"""
ConvI — Speech Pipeline Orchestrator
Module : app/speech_pipeline/pipeline.py

Entry point for the AUDIO path. Handles the 3-stage pipeline:

  Stage 1 — Speaker Diarization  (pyannote.audio)
  Stage 2 — ASR + Lang Detection (faster-whisper)
  Stage 3 — Audio Emotion        (speechbrain wav2vec2)

Supports two audio modes (resolved automatically by language detection
or explicit force):

  Mode 1 : English (.wav)   → forced_language="en"  or auto-detect → "en"
  Mode 2 : Malayalam (.wav) → forced_language="ml"  or auto-detect → "ml"

Final output:  List[SpeechSegment]
  Each SpeechSegment is a fully enriched unit ready for the
  Conversation Normalizer to build a ConversationTurn from.

Usage
-----
from app.speech_pipeline.pipeline import run_speech_pipeline

segments = run_speech_pipeline(
    audio_path="/tmp/call.wav",
    forced_language=None,   # or "en" / "ml"
)
"""

from __future__ import annotations

import asyncio
import os
import shutil
import tempfile
from pathlib import Path
from typing import List, Optional

from loguru import logger

from app.config import get_settings
from app.speech_pipeline.schemas import SpeechSegment
from app.speech_pipeline.diarizer import diarize
from app.speech_pipeline.transcriber import transcribe, detect_audio_language
from app.speech_pipeline.emotion_detector import detect_emotions

# Languages explicitly supported (Whisper ISO-639-1 codes)
SUPPORTED_LANGUAGES = {"en", "ml"}


# ══════════════════════════════════════════════════════════════════════════════
# Main orchestrator — synchronous
# ══════════════════════════════════════════════════════════════════════════════

def run_speech_pipeline(
    audio_path: str | Path,
    forced_language: Optional[str] = None,
    skip_emotion: bool = False,
) -> List[SpeechSegment]:
    """
    Full audio analysis pipeline.

    Parameters
    ----------
    audio_path : str | Path
        Path to the input WAV file (16-bit PCM recommended).
    forced_language : Optional[str]
        "en" or "ml" to bypass auto-detection.
        None → faster-whisper auto-detects from first 30 s.
    skip_emotion : bool
        If True, skips Stage 3 (emotion detection).
        Useful for quick testing or when speechbrain is unavailable.

    Returns
    -------
    List[SpeechSegment]
        Chronologically ordered, fully enriched speech segments.
        Empty list if audio contains no detectable speech.

    Raises
    ------
    FileNotFoundError  : audio_path does not exist.
    RuntimeError       : Any sub-stage fails catastrophically.
    """
    audio_path = Path(audio_path)

    if not audio_path.exists():
        raise FileNotFoundError(f"[SpeechPipeline] Audio not found: {audio_path}")

    logger.info(
        f"[SpeechPipeline] ▶ Starting  |  file='{audio_path.name}'  "
        f"|  forced_lang={forced_language!r}"
    )

    # ── Stage 1: Speaker Diarization ──────────────────────────────────────
    num_speakers = get_settings().diarization_num_speakers
    logger.info(
        "[SpeechPipeline] Stage 1/3 — Speaker Diarization"
        + (f" (num_speakers={num_speakers})" if num_speakers else "")
    )
    diarized_segments = diarize(audio_path, num_speakers=num_speakers)

    if not diarized_segments:
        logger.warning("[SpeechPipeline] No speech segments detected. Returning empty list.")
        return []

    # ── Stage 2: ASR + Language Detection ─────────────────────────────────
    logger.info("[SpeechPipeline] Stage 2/3 — ASR & Language Detection")

    # Validate / normalise forced language
    lang = None
    if forced_language:
        lang = forced_language.lower().strip()
        if lang not in SUPPORTED_LANGUAGES:
            logger.warning(
                f"[SpeechPipeline] Unsupported forced_language '{lang}'. "
                f"Falling back to auto-detect."
            )
            lang = None

    transcribed_segments = transcribe(
        audio_path=audio_path,
        diarized_segments=diarized_segments,
        forced_language=lang,
    )

    # Grab detected language from the first transcribed segment
    detected_language = (
        transcribed_segments[0].language if transcribed_segments else (lang or "en")
    )

    # Filter out empty / silence segments (no text produced)
    transcribed_segments = [s for s in transcribed_segments if s.original_text.strip()]
    if not transcribed_segments:
        logger.warning("[SpeechPipeline] All segments were silent. Returning empty list.")
        return []

    # ── Stage 3: Audio Emotion Detection ──────────────────────────────────
    emotion_results = []
    if not skip_emotion:
        logger.info("[SpeechPipeline] Stage 3/3 — Audio Emotion Detection")
        try:
            emotion_results = detect_emotions(audio_path, transcribed_segments)
        except Exception as exc:
            logger.error(
                f"[SpeechPipeline] Emotion detection failed: {exc} — "
                f"all segments will be labelled 'neutral'."
            )
            from app.speech_pipeline.schemas import EmotionResult
            emotion_results = [
                EmotionResult(emotion="neutral", confidence=0.0)
                for _ in transcribed_segments
            ]
    else:
        from app.speech_pipeline.schemas import EmotionResult
        emotion_results = [
            EmotionResult(emotion=None, confidence=0.0)
            for _ in transcribed_segments
        ]

    # ── Merge: build SpeechSegment list ───────────────────────────────────
    speech_segments: List[SpeechSegment] = []

    for t_seg, e_res in zip(transcribed_segments, emotion_results):
        speech_segments.append(SpeechSegment(
            speaker_id               = t_seg.speaker_id,
            start_time               = t_seg.start_time,
            end_time                 = t_seg.end_time,
            original_text            = t_seg.original_text,
            language                 = t_seg.language,
            emotion                  = e_res.emotion,
            emotion_confidence       = e_res.confidence,
            transcription_confidence = t_seg.transcription_confidence,
            audio_language           = detected_language,
        ))

    logger.info(
        f"[SpeechPipeline] ✅ Done  |  {len(speech_segments)} segments  "
        f"|  lang='{detected_language}'  "
        f"|  speakers={len({s.speaker_id for s in speech_segments})}"
    )

    return speech_segments


# ══════════════════════════════════════════════════════════════════════════════
# Async wrapper (for FastAPI endpoint)
# ══════════════════════════════════════════════════════════════════════════════

async def run_speech_pipeline_async(
    audio_path: str | Path,
    forced_language: Optional[str] = None,
    skip_emotion: bool = False,
) -> List[SpeechSegment]:
    """
    Async wrapper around run_speech_pipeline.

    Offloads the CPU-intensive pipeline to a thread pool so the FastAPI
    event loop is never blocked.
    """
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(
        None,
        lambda: run_speech_pipeline(
            audio_path=audio_path,
            forced_language=forced_language,
            skip_emotion=skip_emotion,
        ),
    )


# ══════════════════════════════════════════════════════════════════════════════
# Temporary file helpers (used by the router to save UploadFile → disk)
# ══════════════════════════════════════════════════════════════════════════════

async def save_upload_to_temp(upload_file) -> Path:
    """
    Save a FastAPI UploadFile to a temporary WAV file on disk.

    Returns the Path to the temp file. Caller is responsible for cleanup
    (call delete_temp_file when done).
    """
    suffix = Path(upload_file.filename or "audio.wav").suffix or ".wav"

    tmp = tempfile.NamedTemporaryFile(
        delete=False,
        suffix=suffix,
        prefix="convi_audio_",
    )
    tmp.close()
    tmp_path = Path(tmp.name)

    content = await upload_file.read()
    tmp_path.write_bytes(content)

    logger.debug(f"[SpeechPipeline] Saved upload → {tmp_path} ({len(content)} bytes)")
    return tmp_path


def delete_temp_file(path: Path) -> None:
    """Safely remove a temporary audio file."""
    try:
        path.unlink(missing_ok=True)
        logger.debug(f"[SpeechPipeline] Deleted temp file: {path}")
    except Exception as exc:
        logger.warning(f"[SpeechPipeline] Could not delete temp file {path}: {exc}")
