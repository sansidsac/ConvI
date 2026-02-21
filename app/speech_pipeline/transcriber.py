"""
ConvI — ASR Transcriber
Module : app/speech_pipeline/transcriber.py

Transcribes diarized audio segments using faster-whisper.

Supported languages (for this project)
---------------------------------------
• "en"  — English
• "ml"  — Malayalam (ISO 639-1)

Strategy
--------
1. faster-whisper transcribes the full audio file as one pass (efficient).
2. Each Whisper word-level timestamp is then mapped to the diarized segments
   produced by pyannote, producing per-speaker text blocks.
3. Language is either AUTO-detected (first 30 s of audio) or FORCED if the
   caller already knows the language.

Model
-----
Controlled via WHISPER_MODEL_SIZE env var (default: "large-v3").
For CPU-only mode, "medium" or "small" is recommended for speed.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import List, Optional

from loguru import logger

from app.speech_pipeline.schemas import DiarizedSegment, TranscribedSegment

# ── Lazy model holder ──────────────────────────────────────────────────────
_WHISPER_MODEL = None


def _load_model():
    """Load (and cache) the faster-whisper model."""
    global _WHISPER_MODEL

    if _WHISPER_MODEL is not None:
        return _WHISPER_MODEL

    try:
        from app.config import get_settings
        model_size = get_settings().whisper_model_size
    except Exception:
        model_size = os.getenv("WHISPER_MODEL_SIZE", "large-v3")

    try:
        from faster_whisper import WhisperModel

        logger.info(
            f"[Transcriber] Loading faster-whisper '{model_size}' on CPU …"
        )
        # compute_type="int8" is the best CPU option — balances speed vs quality
        model = WhisperModel(model_size, device="cpu", compute_type="int8")
        _WHISPER_MODEL = model
        logger.info("[Transcriber] ✅ faster-whisper model loaded.")
        return model

    except Exception as exc:
        logger.error(f"[Transcriber] Failed to load Whisper model: {exc}")
        raise RuntimeError(f"Whisper model load failed: {exc}") from exc


# ── Public API ─────────────────────────────────────────────────────────────

def transcribe(
    audio_path: str | Path,
    diarized_segments: List[DiarizedSegment],
    forced_language: Optional[str] = None,
) -> List[TranscribedSegment]:
    """
    Transcribe audio and map transcript words to diarized segments.

    Parameters
    ----------
    audio_path : str | Path
        Absolute path to the WAV audio file.
    diarized_segments : List[DiarizedSegment]
        Output from the diarizer — defines the time windows per speaker.
    forced_language : Optional[str]
        If supplied ("en" or "ml"), skips auto-detection and forces that
        language. If None, faster-whisper auto-detects from the first 30 s.

    Returns
    -------
    List[TranscribedSegment]
        One entry per diarized segment, enriched with transcribed text,
        language code, and average word-level confidence.
    """
    audio_path = Path(audio_path)

    if not audio_path.exists():
        raise FileNotFoundError(f"[Transcriber] Audio not found: {audio_path}")

    model = _load_model()

    # ── Run Whisper transcription (word-level timestamps) ─────────────────
    logger.info(
        f"[Transcriber] Transcribing '{audio_path.name}' | "
        f"lang={forced_language or 'auto'}"
    )

    try:
        whisper_segments_iter, info = model.transcribe(
            str(audio_path),
            language=forced_language,        # None = auto-detect
            word_timestamps=True,            # needed for speaker mapping
            beam_size=5,
            vad_filter=True,                 # skip silence
            vad_parameters={"min_silence_duration_ms": 500},
        )
        # Materialise the lazy generator once
        whisper_segments = list(whisper_segments_iter)
    except Exception as exc:
        logger.error(f"[Transcriber] Whisper transcription error: {exc}")
        raise RuntimeError(f"Transcription failed: {exc}") from exc

    detected_lang = info.language or forced_language or "en"
    logger.info(
        f"[Transcriber] Detected language: '{detected_lang}' "
        f"(probability: {info.language_probability:.2f})"
    )

    # ── Collect all words with timing ─────────────────────────────────────
    all_words = []  # list of (start, end, word_text, probability)
    for seg in whisper_segments:
        if seg.words:
            for w in seg.words:
                all_words.append((w.start, w.end, w.word, w.probability))

    # ── Map words → diarized segments ─────────────────────────────────────
    transcribed: List[TranscribedSegment] = []

    for d_seg in diarized_segments:
        # Words whose midpoint falls within this diarized window
        seg_words = [
            (start, end, word, prob)
            for (start, end, word, prob) in all_words
            if _midpoint_in_window(start, end, d_seg.start_time, d_seg.end_time)
        ]

        if seg_words:
            text = " ".join(w[2].strip() for w in seg_words).strip()
            avg_conf = sum(w[3] for w in seg_words) / len(seg_words)
        else:
            # No words mapped — segment is likely silence or very short
            text = ""
            avg_conf = 0.0

        transcribed.append(TranscribedSegment(
            speaker_id=d_seg.speaker_id,
            start_time=d_seg.start_time,
            end_time=d_seg.end_time,
            original_text=text,
            language=detected_lang,
            transcription_confidence=round(avg_conf, 4),
        ))

    logger.info(
        f"[Transcriber] ✅ {len(transcribed)} segments transcribed | "
        f"lang='{detected_lang}'"
    )
    return transcribed


def detect_audio_language(audio_path: str | Path) -> str:
    """
    Lightweight language detection using faster-whisper (first 30 s only).

    Returns
    -------
    str : ISO-639-1 language code (e.g. "en", "ml").
    """
    audio_path = Path(audio_path)
    model = _load_model()

    try:
        _, info = model.transcribe(
            str(audio_path),
            language=None,
            word_timestamps=False,
            beam_size=1,
            # Whisper internally only uses first 30 s for detection
        )
        lang = info.language or "en"
        logger.info(
            f"[Transcriber] Language detection: '{lang}' "
            f"(p={info.language_probability:.2f})"
        )
        return lang
    except Exception as exc:
        logger.warning(f"[Transcriber] Language detection failed: {exc}. Defaulting to 'en'.")
        return "en"


# ── Helpers ────────────────────────────────────────────────────────────────

def _midpoint_in_window(
    word_start: float,
    word_end: float,
    window_start: float,
    window_end: float,
) -> bool:
    """
    Return True if the word's midpoint timestamp falls within the
    [window_start, window_end] interval (inclusive).

    Using midpoint-based assignment avoids double-counting words that
    straddle segment boundaries.
    """
    midpoint = (word_start + word_end) / 2.0
    return window_start <= midpoint <= window_end
