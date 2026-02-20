"""
ConvI — Application Configuration
Reads settings from environment variables / .env file.
"""

from pydantic_settings import BaseSettings, SettingsConfigDict
from functools import lru_cache


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # ── App ──────────────────────────────────────────────────
    app_name: str = "ConvI — Conversation Intelligence API"
    app_version: str = "0.1.0"
    debug: bool = False

    # ── Domain ───────────────────────────────────────────────
    default_domain: str = "financial_banking"

    # ── PostgreSQL ───────────────────────────────────────────
    database_url: str = "postgresql://convI_user:password@localhost:5432/convi"

    # ── FAISS ────────────────────────────────────────────────
    faiss_index_path: str = "data/faiss_index"

    # ── Models ───────────────────────────────────────────────
    whisper_model_size: str = "large-v3"
    spacy_model: str = "en_core_web_sm"

    # ── Speech Pipeline ──────────────────────────────────────
    # HuggingFace token required for pyannote gated models.
    # Without it, diarization falls back to single-speaker mode.
    pyannote_auth_token: str = ""
    # Force number of speakers (e.g. 2 for agent+customer). None = auto-detect.
    diarization_num_speakers: int | None = 2
    # SpeechBrain emotion recognition model (HuggingFace hub id)
    emotion_model: str = "speechbrain/emotion-recognition-wav2vec2-IEMOCAP"
    # ISO-639-1 codes of audio languages supported by the pipeline
    supported_audio_languages: list[str] = ["en", "ml"]

    # ── bge-m3 ───────────────────────────────────────────────
    embedding_model: str = "BAAI/bge-m3"

    # ── Local LLM (Qwen2.5) ──────────────────────────────────
    llm_model_path: str = "Qwen/Qwen2.5-7B-Instruct"
    llm_max_new_tokens: int = 1024


@lru_cache
def get_settings() -> Settings:
    """Cache-backed settings loader."""
    return Settings()
