"""
ConvI — Audio Emotion Detector
Module : app/speech_pipeline/emotion_detector.py

Classifies the emotion of each audio segment using a pretrained
SpeechBrain wav2vec2 model fine-tuned on IEMOCAP.

Model
-----
speechbrain/emotion-recognition-wav2vec2-IEMOCAP
• Emotion classes: neutral (neu), happy (hap), angry (ang), sad (sad)
• Input: 16-kHz mono WAV (the model handles resampling internally via
  torchaudio if the source sample-rate differs)
• Each diarized segment is sliced from the full audio tensor in memory
  (no disk I/O per segment).

CPU note
--------
wav2vec2 is compute-intensive. On CPU this can be slow for long calls.
For hackathon purposes this is acceptable; GPU would accelerate ~10×.

Output contract  →  EmotionResult per segment
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import List, Optional, Tuple

import torch
import torchaudio
from loguru import logger

from app.speech_pipeline.schemas import DiarizedSegment, EmotionResult, TranscribedSegment

# ── Constants ──────────────────────────────────────────────────────────────
_MODEL_NAME    = "speechbrain/emotion-recognition-wav2vec2-IEMOCAP"
_TARGET_SR     = 16_000   # model expects 16 kHz
_MIN_DURATION  = 0.5      # seconds — skip segments shorter than this

# Map SpeechBrain IEMOCAP codes → human-readable labels
_LABEL_MAP = {
    "neu": "neutral",
    "hap": "happy",
    "ang": "angry",
    "sad": "sad",
    # wav2vec2-IEMOCAP does NOT output fear/disgust, but guard anyway
    "fea": "fear",
    "dis": "disgust",
}

# ── Lazy model holder ──────────────────────────────────────────────────────
_EMOTION_CLASSIFIER = None


def _load_classifier():
    """Load (and cache) the SpeechBrain emotion classifier."""
    global _EMOTION_CLASSIFIER

    if _EMOTION_CLASSIFIER is not None:
        return _EMOTION_CLASSIFIER

    try:
        from speechbrain.inference.interfaces import foreign_class

        logger.info(
            f"[EmotionDetector] Loading '{_MODEL_NAME}' on CPU …"
        )
        # SpeechBrain downloads to ~/.cache/huggingface on first run
        classifier = foreign_class(
            source=_MODEL_NAME,
            pymodule_file="custom_interface.py",
            classname="CustomEncoderWav2vec2Classifier",
            run_opts={"device": "cpu"},
        )
        _EMOTION_CLASSIFIER = classifier
        logger.info("[EmotionDetector] ✅ Emotion classifier loaded.")
        return classifier

    except Exception as exc:
        logger.error(f"[EmotionDetector] Failed to load model: {exc}")
        raise RuntimeError(f"Emotion model load failed: {exc}") from exc


# ── Public API ─────────────────────────────────────────────────────────────

def detect_emotions(
    audio_path: str | Path,
    segments: List[DiarizedSegment] | List[TranscribedSegment],
) -> List[EmotionResult]:
    """
    Detect emotion for each diarized / transcribed segment.

    Parameters
    ----------
    audio_path : str | Path
        Path to the full WAV audio file.
    segments : List[DiarizedSegment | TranscribedSegment]
        Each segment provides .start_time and .end_time in seconds.

    Returns
    -------
    List[EmotionResult]
        One EmotionResult per incoming segment, in the same order.
        Segments that are too short or fail inference get emotion="neutral"
        with confidence=0.0.
    """
    audio_path = Path(audio_path)
    if not audio_path.exists():
        raise FileNotFoundError(f"[EmotionDetector] File not found: {audio_path}")

    classifier = _load_classifier()

    # Load full audio once → slice per segment in-memory
    waveform, sr = _load_wav(audio_path)
    waveform = _to_16k_mono(waveform, sr)  # normalise to model requirements

    results: List[EmotionResult] = []

    for i, seg in enumerate(segments):
        duration = seg.end_time - seg.start_time

        # Skip very short segments (likely silence / noise)
        if duration < _MIN_DURATION:
            logger.debug(
                f"[EmotionDetector] Segment {i} too short "
                f"({duration:.2f}s) → defaulting to 'neutral'"
            )
            results.append(EmotionResult(emotion="neutral", confidence=0.0))
            continue

        # Slice the segment waveform
        seg_wave = _slice_waveform(waveform, seg.start_time, seg.end_time, _TARGET_SR)

        try:
            emotion, confidence = _infer_emotion(classifier, seg_wave)
            results.append(EmotionResult(emotion=emotion, confidence=confidence))
            logger.debug(
                f"[EmotionDetector] Seg {i} "
                f"[{seg.start_time:.1f}s–{seg.end_time:.1f}s] "
                f"→ '{emotion}' (conf={confidence:.2f})"
            )
        except Exception as exc:
            logger.warning(
                f"[EmotionDetector] Seg {i} inference failed: {exc} "
                f"→ defaulting to 'neutral'"
            )
            results.append(EmotionResult(emotion="neutral", confidence=0.0))

    logger.info(
        f"[EmotionDetector] ✅ Emotions detected for {len(results)} segments."
    )
    return results


# ── Inference helper ───────────────────────────────────────────────────────

def _infer_emotion(
    classifier,
    waveform: torch.Tensor,
) -> Tuple[str, float]:
    """
    Run the SpeechBrain emotion classifier on a single waveform tensor.

    Returns
    -------
    Tuple[str, float]
        (human-readable emotion label, confidence score)
    """
    # SpeechBrain's foreign_class classifier expects a batch tensor
    # shape: (batch=1, time)
    if waveform.dim() == 1:
        waveform = waveform.unsqueeze(0)

    relative_lengths = torch.tensor([1.0])

    with torch.no_grad():
        out_prob, score, index, label = classifier.classify_batch(
            waveform, relative_lengths
        )

    raw_label: str = label[0] if label[0] is not None else "neu"
    confidence = float(score[0].item())
    human_label = _LABEL_MAP.get(raw_label, raw_label)
    return human_label, confidence


# ── Audio helpers ──────────────────────────────────────────────────────────

def _load_wav(path: Path) -> Tuple[torch.Tensor, int]:
    """Load a WAV file and return (waveform, sample_rate)."""
    waveform, sr = torchaudio.load(str(path))
    return waveform, sr


def _to_16k_mono(waveform: torch.Tensor, sr: int) -> torch.Tensor:
    """
    Resample to 16 kHz and mix down to mono if necessary.
    Returns tensor of shape (time,).
    """
    # Mix-down multi-channel to mono
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)

    # Resample if needed
    if sr != _TARGET_SR:
        resampler = torchaudio.transforms.Resample(
            orig_freq=sr, new_freq=_TARGET_SR
        )
        waveform = resampler(waveform)

    return waveform.squeeze(0)  # shape: (time,)


def _slice_waveform(
    waveform: torch.Tensor,
    start_s: float,
    end_s: float,
    sr: int,
) -> torch.Tensor:
    """
    Slice [start_s, end_s] from a 1-D waveform tensor.
    Clamps indices to valid range to avoid out-of-bounds errors.
    """
    total_samples = waveform.shape[0]
    start_idx = min(int(start_s * sr), total_samples - 1)
    end_idx   = min(int(end_s   * sr), total_samples)
    end_idx   = max(end_idx, start_idx + 1)  # guarantee at least 1 sample
    return waveform[start_idx:end_idx]
