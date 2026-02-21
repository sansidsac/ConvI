"""
ConvI — Model Pre-downloader
=============================
Run this ONCE before starting the server to download all heavy ML models
to their proper local caches. The server will then load instantly.

Usage:
    venv\Scripts\python download_models.py     ← Windows (always use venv!)

Models downloaded:
    1. faster-whisper large-v3     (~3.1 GB) → HuggingFace cache
    2. BAAI/bge-m3                 (~2.3 GB) → HuggingFace cache
    3. pyannote/segmentation-3.0         (needs HF token) → models/
    4. pyannote/wespeaker-voxceleb       (needs HF token) → models/
    5. pyannote/speaker-diarization-3.1  (needs HF token) → models/

Note:
    SpeechBrain emotion model auto-downloads to HuggingFace cache
    on first server request — no manual pre-download needed.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from loguru import logger

# ── project root → add to path ────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

# ── Config ────────────────────────────────────────────────────────────────────
from app.config import get_settings
settings = get_settings()

HF_TOKEN     = settings.pyannote_auth_token or os.getenv("HF_TOKEN", "")
MODELS_DIR   = ROOT / "models"
MODELS_DIR.mkdir(exist_ok=True)



# ═══════════════════════════════════════════════════════════════════════════════
# 1. faster-whisper (large-v3)
# ═══════════════════════════════════════════════════════════════════════════════

def download_whisper():
    model_size = settings.whisper_model_size
    logger.info(f"[1/5] faster-whisper '{model_size}'  (~3.1 GB) ...")
    try:
        from faster_whisper import WhisperModel
        model = WhisperModel(model_size, device="cpu", compute_type="int8")
        logger.success(f"    ✅ faster-whisper '{model_size}' ready.")
        del model
    except Exception as e:
        logger.error(f"    ✗  faster-whisper failed: {e}")


# ═══════════════════════════════════════════════════════════════════════════════
# 2. BAAI/bge-m3  (embedding model for RAG)
# ═══════════════════════════════════════════════════════════════════════════════

def download_bge_m3():
    logger.info("[2/5] BAAI/bge-m3  (~2.3 GB) ...")
    try:
        from FlagEmbedding import FlagModel
        m = FlagModel("BAAI/bge-m3", use_fp16=False)
        _ = m.encode(["test"])
        logger.success("    ✅ bge-m3 ready.")
        del m
    except Exception as e:
        logger.error(f"    ✗  bge-m3 failed: {e}")


# ═══════════════════════════════════════════════════════════════════════════════
# 3-5. pyannote models  (gated — need HF token)
# ═══════════════════════════════════════════════════════════════════════════════

def download_pyannote():
    if not HF_TOKEN:
        logger.warning(
            "[3-5/5] PYANNOTE_AUTH_TOKEN not set — skipping pyannote download.\n"
            "        Set PYANNOTE_AUTH_TOKEN in .env or export HF_TOKEN to pre-download."
        )
        return

    from huggingface_hub import snapshot_download

    pyannote_models = [
        ("pyannote/segmentation-3.0",              MODELS_DIR / "segmentation-3.0"),
        ("pyannote/wespeaker-voxceleb-resnet34-LM", MODELS_DIR / "wespeaker-voxceleb-resnet34-LM"),
        ("pyannote/speaker-diarization-3.1",        MODELS_DIR / "speaker-diarization-3.1"),
    ]

    for i, (repo_id, dest) in enumerate(pyannote_models, start=3):
        if dest.exists() and any(dest.iterdir()):
            logger.info(f"[{i}/6] {repo_id} — already cached at {dest}")
            continue

        logger.info(f"[{i}/6] Downloading {repo_id} → {dest} ...")
        try:
            snapshot_download(
                repo_id=repo_id,
                token=HF_TOKEN,
                local_dir=str(dest),
                local_dir_use_symlinks=False,
            )
            logger.success(f"    ✅ {repo_id} ready.")
        except Exception as e:
            logger.error(f"    ✗  {repo_id} failed: {e}")

    # Patch config.yaml so pyannote uses local model paths
    dia_config = MODELS_DIR / "speaker-diarization-3.1" / "config.yaml"
    if dia_config.exists():
        seg_dir   = MODELS_DIR / "segmentation-3.0"
        embed_dir_py = MODELS_DIR / "pyannote_wespeaker-voxceleb-resnet34-LM"
        embed_dir    = MODELS_DIR / "wespeaker-voxceleb-resnet34-LM"

        # Copy wespeaker to pyannote-prefixed dir
        embed_dir_py.mkdir(parents=True, exist_ok=True)
        embed_bin_py = embed_dir_py / "pytorch_model.bin"
        if not embed_bin_py.exists() and (embed_dir / "pytorch_model.bin").exists():
            import shutil
            shutil.copy2(embed_dir / "pytorch_model.bin", embed_bin_py)
            logger.info("    Copied wespeaker → pyannote_wespeaker (PyTorch loader fix).")

        text = dia_config.read_text(encoding="utf-8")
        replacements = {
            "pyannote/segmentation-3.0":               str(seg_dir).replace("\\", "/"),
            "pyannote/wespeaker-voxceleb-resnet34-LM": str(embed_dir_py).replace("\\", "/"),
        }
        changed = False
        for old, new in replacements.items():
            if old in text:
                text = text.replace(old, new)
                changed = True
        if changed:
            dia_config.write_text(text, encoding="utf-8")
            logger.info("    config.yaml patched with local model paths.")


# ═══════════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    logger.info("=" * 60)
    logger.info("ConvI — Pre-downloading all models")
    logger.info("=" * 60)

    download_whisper()
    download_bge_m3()
    download_pyannote()

    logger.success("\n🎉 All downloads complete. You can now start the server:")
    logger.success("   venv\\Scripts\\uvicorn app.main:app --host 0.0.0.0 --port 8000")
    logger.info("   Note: SpeechBrain emotion model will auto-download on first request.")
