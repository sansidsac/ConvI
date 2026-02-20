"""
ConvI - Speaker Diarizer
Module : app/speech_pipeline/diarizer.py

Performs speaker diarization on a WAV audio file using pyannote.audio 3.x.

Windows-specific notes
-----------------------
* HuggingFace Hub's default cache uses symlinks, which Windows forbids without
  Developer Mode.  We download both models to a flat local directory
  (``models/`` in the project root) using ``local_dir_use_symlinks=False``.
* pyannote loads segmentation-3.0 dynamically from config.yaml; we patch that
  YAML to point at the local directory so no network call is needed at runtime.
* torchaudio 2.x removed list_audio_backends / get_audio_backend; we monkey-
  patch them before importing pyannote so it doesn't raise AttributeError.
* torchcodec (built-in audio decoder) is broken on this install; we pre-load
  audio with torchaudio and pass a waveform dict directly to pyannote.

Output contract  -->  List[DiarizedSegment]
"""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import List, Optional

from loguru import logger

from app.speech_pipeline.schemas import DiarizedSegment

# ── Lazy model holder ──────────────────────────────────────────────────────
_DIARIZATION_PIPELINE = None  # loaded on first call

# Project-local directory where both pyannote models are downloaded
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
_MODELS_DIR   = _PROJECT_ROOT / "models"


# ══════════════════════════════════════════════════════════════════════════
# Internal helpers
# ══════════════════════════════════════════════════════════════════════════

def _apply_torchaudio_patches() -> None:
    """
    Monkey-patch APIs removed in torchaudio 2.x so pyannote doesn't crash on
    import.  This is safe to call multiple times.
    """
    import torchaudio as _ta
    import warnings
    warnings.filterwarnings("ignore", message="torchcodec is not installed")
    warnings.filterwarnings("ignore", category=UserWarning, module="pyannote")

    if not hasattr(_ta, "list_audio_backends"):
        _ta.list_audio_backends = lambda: ["soundfile"]
    if not hasattr(_ta, "get_audio_backend"):
        _ta.get_audio_backend = lambda: "soundfile"
    if not hasattr(_ta, "set_audio_backend"):
        _ta.set_audio_backend = lambda _b: None


def _download_model(repo_id: str, dest: Path, token: str) -> bool:
    """
    Download a HuggingFace model repo to a flat local directory with no
    symlinks.  Returns True on success, False on failure.
    """
    from huggingface_hub import snapshot_download

    try:
        logger.info(f"[Diarizer] Downloading {repo_id} -> {dest} ...")
        snapshot_download(
            repo_id=repo_id,
            token=token,
            local_dir=str(dest),
            local_dir_use_symlinks=False,   # key fix for Windows
        )
        logger.info(f"[Diarizer] Downloaded {repo_id} OK.")
        return True
    except Exception as exc:
        logger.error(f"[Diarizer] Failed to download {repo_id}: {exc}")
        return False


def _ensure_models(token: str) -> Optional[Path]:
    """
    Ensure all three pyannote models exist locally.
    Downloads to flat directories under models/ with no symlinks.
    Patches config.yaml to use local paths for all sub-model references.
    Returns path to speaker-diarization-3.1, or None on failure.
    """
    dia_dir     = _MODELS_DIR / "speaker-diarization-3.1"
    seg_dir     = _MODELS_DIR / "segmentation-3.0"
    embed_dir   = _MODELS_DIR / "wespeaker-voxceleb-resnet34-LM"
    # Path with "pyannote" in name — forces PyTorch loader instead of ONNX
    # (fixes ONNXRuntimeError system error 13 on Windows; see pyannote#1660)
    embed_dir_py = _MODELS_DIR / "pyannote_wespeaker-voxceleb-resnet34-LM"

    # Sub-models pyannote needs (must download before loading the pipeline)
    sub_models = [
        ("pyannote/segmentation-3.0",                  seg_dir,   "pytorch_model.bin"),
        ("pyannote/wespeaker-voxceleb-resnet34-LM",    embed_dir, "pytorch_model.bin"),
    ]

    for repo_id, dest, sentinel in sub_models:
        if not (dest / sentinel).exists():
            ok = _download_model(repo_id, dest, token)
            if not ok:
                return None
        else:
            logger.debug(f"[Diarizer] {repo_id} already cached at {dest}")

    # Copy wespeaker to pyannote-prefixed dir so PyTorch loader is used (not ONNX)
    embed_dir_py.mkdir(parents=True, exist_ok=True)
    embed_bin_py = embed_dir_py / "pytorch_model.bin"
    if not embed_bin_py.exists():
        import shutil
        shutil.copy2(embed_dir / "pytorch_model.bin", embed_bin_py)
        logger.info("[Diarizer] Created pyannote_wespeaker copy for PyTorch loader.")

    # Main pipeline
    dia_config = dia_dir / "config.yaml"
    if not dia_config.exists():
        ok = _download_model("pyannote/speaker-diarization-3.1", dia_dir, token)
        if not ok:
            return None
    else:
        logger.debug(f"[Diarizer] speaker-diarization-3.1 already cached at {dia_dir}")

    # Patch config.yaml — use pyannote-prefixed wespeaker path (PyTorch, not ONNX)
    try:
        config_text = dia_config.read_text(encoding="utf-8")
        embed_path_py = str(embed_dir_py).replace("\\", "/")
        embed_path_old = str(embed_dir).replace("\\", "/")
        replacements = {
            "pyannote/segmentation-3.0":               str(seg_dir).replace("\\", "/"),
            "pyannote/wespeaker-voxceleb-resnet34-LM": embed_path_py,
            embed_path_old:                            embed_path_py,  # fix already-patched config
        }
        changed = False
        for old_val, new_val in replacements.items():
            if old_val in config_text and old_val != new_val:
                config_text = config_text.replace(old_val, new_val)
                changed = True
        if changed:
            dia_config.write_text(config_text, encoding="utf-8")
            logger.info("[Diarizer] config.yaml patched with local model paths.")
        else:
            logger.debug("[Diarizer] config.yaml already uses local paths.")
    except Exception as exc:
        logger.warning(f"[Diarizer] Could not patch config.yaml: {exc}")

    return dia_dir


def _load_pipeline() -> Optional[object]:
    """Load (and cache) the pyannote diarization pipeline from a local directory."""
    global _DIARIZATION_PIPELINE

    if _DIARIZATION_PIPELINE is not None:
        return _DIARIZATION_PIPELINE

    # Prefer app config so .env is single source of truth
    try:
        from app.config import get_settings
        auth_token = (get_settings().pyannote_auth_token or "").strip()
    except Exception:
        auth_token = os.getenv("PYANNOTE_AUTH_TOKEN", "").strip()
    if not auth_token:
        logger.warning(
            "[Diarizer] PYANNOTE_AUTH_TOKEN not set - "
            "running in SINGLE-SPEAKER fallback mode."
        )
        return None

    try:
        _apply_torchaudio_patches()
        import torch
        from pyannote.audio import Pipeline

        # Ensure huggingface_hub uses token for nested fetches (e.g. speaker-diarization-community-1)
        os.environ["HF_TOKEN"] = auth_token
        os.environ["HUGGING_FACE_HUB_TOKEN"] = auth_token

        # Ensure models are on disk
        dia_dir = _ensure_models(auth_token)
        if dia_dir is None:
            logger.error("[Diarizer] Could not download models. Falling back.")
            return None

        logger.info(f"[Diarizer] Loading pipeline from {dia_dir} ...")
        # Token required: pipeline fetches xvec_transform from speaker-diarization-community-1
        pipeline = Pipeline.from_pretrained(
            str(dia_dir),
            token=auth_token,
        )
        pipeline.to(torch.device("cpu"))

        _DIARIZATION_PIPELINE = pipeline
        logger.info("[Diarizer] pyannote pipeline loaded on CPU.")
        return pipeline

    except Exception as exc:
        err_str = str(exc)
        if "401" in err_str or "403" in err_str or "gated" in err_str.lower():
            logger.error(
                "[Diarizer] HuggingFace access denied.\n"
                "  Accept terms at:\n"
                "  1. https://huggingface.co/pyannote/speaker-diarization-3.1\n"
                "  2. https://huggingface.co/pyannote/segmentation-3.0\n"
                "  3. https://huggingface.co/pyannote/speaker-diarization-community-1"
            )
        else:
            logger.error(f"[Diarizer] Failed to load pyannote pipeline: {exc}")
        logger.warning("[Diarizer] Falling back to SINGLE-SPEAKER mode.")
        return None


# ══════════════════════════════════════════════════════════════════════════
# Public API
# ══════════════════════════════════════════════════════════════════════════

def diarize(audio_path: str | Path, num_speakers: Optional[int] = None) -> List[DiarizedSegment]:
    """
    Run speaker diarization on *audio_path* (WAV file).

    Parameters
    ----------
    audio_path : str | Path
        Absolute path to the WAV audio file.
    num_speakers : Optional[int]
        If set (e.g. 2 for agent+customer), constrains diarization to this many
        speakers. Omit for automatic estimation.

    Returns
    -------
    List[DiarizedSegment]
        Chronologically sorted list of speaker segments.
        Falls back to a single SPEAKER_00 segment if pyannote is unavailable.
    """
    audio_path = Path(audio_path)

    if not audio_path.exists():
        raise FileNotFoundError(f"[Diarizer] Audio file not found: {audio_path}")

    pipeline = _load_pipeline()

    # Fallback: single speaker covering the full audio
    if pipeline is None:
        duration = _get_wav_duration(audio_path)
        logger.info(
            f"[Diarizer] Single-speaker fallback -> SPEAKER_00 for {duration:.2f}s"
        )
        return [DiarizedSegment(
            speaker_id="SPEAKER_00",
            start_time=0.0,
            end_time=duration,
        )]

    # Real diarization - pre-load audio as tensor to bypass broken torchcodec
    logger.info(
        f"[Diarizer] Running diarization on: {audio_path.name}"
        + (f" (num_speakers={num_speakers})" if num_speakers is not None else "")
    )
    try:
        import soundfile as sf
        import torch
        data, sample_rate = sf.read(str(audio_path), dtype="float32")
        if data.ndim == 1:
            data = data[None, :]  # (samples,) -> (1, samples)
        elif data.ndim == 2 and data.shape[0] > data.shape[1]:
            data = data.T  # (samples, ch) -> (ch, samples)
        waveform = torch.from_numpy(data)
        logger.debug(
            f"[Diarizer] Audio pre-loaded: shape={waveform.shape}, sr={sample_rate}"
        )
        audio_dict = {"waveform": waveform, "sample_rate": sample_rate}
        kwargs = {} if num_speakers is None else {"num_speakers": num_speakers}
        diarization = pipeline(audio_dict, **kwargs)
    except Exception as exc:
        logger.error(f"[Diarizer] Diarization inference error: {exc}")
        raise RuntimeError(f"Diarization failed: {exc}") from exc

    segments: List[DiarizedSegment] = []
    # pyannote 3.x: itertracks; pyannote 4.x: speaker_diarization
    if hasattr(diarization, "speaker_diarization"):
        for turn, speaker in diarization.speaker_diarization:
            segments.append(DiarizedSegment(
                speaker_id=str(speaker),
                start_time=round(turn.start, 3),
                end_time=round(turn.end, 3),
            ))
    else:
        for turn, _, speaker in diarization.itertracks(yield_label=True):
            segments.append(DiarizedSegment(
                speaker_id=speaker,
                start_time=round(turn.start, 3),
                end_time=round(turn.end, 3),
            ))

    segments.sort(key=lambda s: s.start_time)

    logger.info(
        f"[Diarizer] {len(segments)} segments | "
        f"{len({s.speaker_id for s in segments})} speaker(s)"
    )
    return segments


# ══════════════════════════════════════════════════════════════════════════
# Helpers
# ══════════════════════════════════════════════════════════════════════════

def _get_wav_duration(path: Path) -> float:
    """Return WAV duration in seconds using stdlib wave (no extra deps)."""
    import wave
    try:
        with wave.open(str(path), "rb") as wf:
            return wf.getnframes() / float(wf.getframerate())
    except Exception:
        return 60.0
