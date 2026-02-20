"""
ConvI — Speech Pipeline Module
===============================
Public API surface for app.speech_pipeline.

Exports
-------
run_speech_pipeline        — synchronous orchestrator
run_speech_pipeline_async  — async wrapper (FastAPI-friendly)
save_upload_to_temp        — save UploadFile → temp WAV
delete_temp_file           — cleanup temp WAV
SpeechSegment              — output schema consumed by the Normalizer
SUPPORTED_LANGUAGES        — {"en", "ml"}
"""

from app.speech_pipeline.pipeline import (
    run_speech_pipeline,
    run_speech_pipeline_async,
    save_upload_to_temp,
    delete_temp_file,
    SUPPORTED_LANGUAGES,
)
from app.speech_pipeline.schemas import SpeechSegment

__all__ = [
    "run_speech_pipeline",
    "run_speech_pipeline_async",
    "save_upload_to_temp",
    "delete_temp_file",
    "SUPPORTED_LANGUAGES",
    "SpeechSegment",
]
