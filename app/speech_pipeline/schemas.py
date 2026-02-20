"""
ConvI — Speech Pipeline Internal Schemas

Lightweight dataclasses used as internal contracts between the three
speech pipeline sub-modules (diarizer → transcriber → emotion_detector).
These are NOT the public API schemas (those live in app/schemas/__init__.py).
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional


# ── Diarizer output ────────────────────────────────────────────────────────

@dataclass
class DiarizedSegment:
    """
    A single speaker turn produced by the diarization stage.

    Attributes:
        speaker_id : Anonymized label  e.g. "SPEAKER_00", "SPEAKER_01"
        start_time : Segment start in seconds (from audio beginning).
        end_time   : Segment end in seconds.
    """
    speaker_id: str
    start_time: float
    end_time:   float


# ── Transcriber output ─────────────────────────────────────────────────────

@dataclass
class TranscribedSegment:
    """
    Enriches a DiarizedSegment with ASR text and language info.

    Attributes:
        speaker_id          : Inherited from DiarizedSegment.
        start_time          : Segment start in seconds.
        end_time            : Segment end in seconds.
        original_text       : Raw transcription in source language.
        language            : ISO-639-1 code detected / forced ("en", "ml").
        transcription_confidence : Average word-level probability [0, 1].
    """
    speaker_id:                str
    start_time:                float
    end_time:                  float
    original_text:             str
    language:                  str
    transcription_confidence:  float = 1.0


# ── Emotion detector output ────────────────────────────────────────────────

@dataclass
class EmotionResult:
    """
    Emotion label and confidence for a single audio segment.

    Attributes:
        emotion    : Predicted emotion label
                     ("neutral", "happy", "angry", "sad", "fear", "disgust").
        confidence : Model confidence for the winning label [0, 1].
    """
    emotion:    str
    confidence: float


# ── Final enriched segment (pipeline output unit) ──────────────────────────

@dataclass
class SpeechSegment:
    """
    Fully enriched segment produced by the speech pipeline orchestrator.
    This is the object handed off to the Conversation Normalizer.

    Attributes:
        speaker_id               : e.g. "SPEAKER_00"
        start_time               : Segment start (seconds).
        end_time                 : Segment end (seconds).
        original_text            : Transcribed text in source language.
        language                 : ISO-639-1 source language code.
        emotion                  : Predicted emotion (may be None on error).
        emotion_confidence       : Confidence for the emotion label.
        transcription_confidence : Average ASR word probability.
        audio_language           : Language detected at the file level
                                   ("en" | "ml") — same across all segments.
    """
    speaker_id:                str
    start_time:                float
    end_time:                  float
    original_text:             str
    language:                  str
    emotion:                   Optional[str]  = None
    emotion_confidence:        float          = 0.0
    transcription_confidence:  float          = 1.0
    audio_language:            str            = "en"
