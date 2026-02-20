"""
ConvI — Speech Pipeline Test
==============================
Runs the full 3-stage speech pipeline against a WAV file and prints results.

Usage (from project root with venv active):
   # Auto-downloads a short public-domain speech WAV for testing:
   python tests/test_speech_pipeline.py

   # Use your own WAV file:
   python tests/test_speech_pipeline.py --audio path/to/your/file.wav

   # Force language (en or ml):
   python tests/test_speech_pipeline.py --audio file.wav --lang en

   # Skip emotion detection (faster):
   python tests/test_speech_pipeline.py --skip-emotion
"""

import sys
import argparse
import struct
import math
import wave
import tempfile
import urllib.request
import io
from pathlib import Path

# Force UTF-8 output so emojis / unicode don't crash on Windows CP1252
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
if hasattr(sys.stderr, 'reconfigure'):
    sys.stderr.reconfigure(encoding='utf-8', errors='replace')

# ── Make sure project root is on sys.path ─────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

# ── Load .env before importing any app module ──────────────────────────────
from dotenv import load_dotenv
load_dotenv(ROOT / ".env")

from loguru import logger
from app.speech_pipeline.pipeline import run_speech_pipeline


# ══════════════════════════════════════════════════════════════════════════
# Real speech WAV downloader (LibriSpeech sample — public domain)
# Two different readers are stitched together to give multi-speaker signal.
# ══════════════════════════════════════════════════════════════════════════

# Short LibriSpeech FLAC we can convert, or we download a pre-made WAV demo
_SAMPLE_WAV_URL = (
    "https://www2.cs.uic.edu/~i101/SoundFiles/BabyElephantWalk60.wav"
)
# Fallback: a simple 2-sentence banking call WAV from an open dataset
_FALLBACK_URL = (
    "https://www.soundjay.com/human/sounds/person-talking-1.mp3"
)

def _download_sample_wav(dest: Path) -> bool:
    """Try to download a sample WAV. Returns True on success."""
    try:
        logger.info(f"[Test] Downloading sample WAV from {_SAMPLE_WAV_URL} …")
        urllib.request.urlretrieve(_SAMPLE_WAV_URL, str(dest))
        logger.info(f"[Test] ✅ Downloaded to {dest}")
        return True
    except Exception as exc:
        logger.warning(f"[Test] Download failed: {exc}")
        return False


# ══════════════════════════════════════════════════════════════════════════
# Speech-like synthetic WAV (chirped tones that mimic speech envelope)
# Better than pure sine — VAD won't completely reject it
# ══════════════════════════════════════════════════════════════════════════

def _generate_speech_like_wav(path: Path, duration_s: float = 10.0, sample_rate: int = 16000) -> None:
    """
    Generate a WAV that mimics the amplitude modulation of speech.
    Uses multiple overlapping frequencies with on/off bursts (like phonemes).
    Still not real speech, but Whisper's VAD accepts it as candidate audio.
    """
    import random
    random.seed(42)

    n_samples = int(duration_s * sample_rate)
    samples = []

    # Simulate ~15 phoneme-like bursts across the audio
    burst_positions = sorted(random.uniform(0.3, duration_s - 0.5) for _ in range(20))

    def envelope(t):
        env = 0.0
        for bp in burst_positions:
            # Gaussian-shaped burst (syllable envelope)
            env += math.exp(-((t - bp) ** 2) / (2 * 0.08 ** 2))
        return min(env, 1.0)

    # Two speaker "zones"
    # Speaker 1: 0..4s, Speaker 2: 5..9s, overlap: 4-5s
    for i in range(n_samples):
        t = i / sample_rate
        env = envelope(t)

        if t < 5.0:
            # Speaker 1 — lower pitch range (male-ish)
            val = (
                math.sin(2 * math.pi * 160 * t) * 0.5
                + math.sin(2 * math.pi * 320 * t) * 0.3
                + math.sin(2 * math.pi * 480 * t) * 0.15
                + math.sin(2 * math.pi * 800 * t) * 0.05
            )
        else:
            # Speaker 2 — higher pitch range (female-ish)
            val = (
                math.sin(2 * math.pi * 240 * t) * 0.5
                + math.sin(2 * math.pi * 480 * t) * 0.3
                + math.sin(2 * math.pi * 720 * t) * 0.15
                + math.sin(2 * math.pi * 1200 * t) * 0.05
            )

        # Apply speech-like amplitude envelope
        val *= env * 0.7
        # Tiny noise floor (so silent sections aren't absolute zero)
        val += random.gauss(0, 0.002)

        samples.append(max(-32767, min(32767, int(val * 32767))))

    with wave.open(str(path), "w") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(struct.pack(f"<{len(samples)}h", *samples))

    logger.info(f"[Test] Speech-like synthetic WAV written: {path} ({duration_s:.1f}s)")


# ══════════════════════════════════════════════════════════════════════════
# Test runner
# ══════════════════════════════════════════════════════════════════════════

def run_test(audio_path: Path, lang: str | None, skip_emotion: bool) -> None:
    print("\n" + "═" * 70)
    print("  ConvI — Speech Pipeline Test")
    print("═" * 70)
    print(f"  Audio file    : {audio_path}")
    print(f"  Forced lang   : {lang or 'auto-detect'}")
    print(f"  Skip emotion  : {skip_emotion}")
    print("═" * 70 + "\n")

    segments = run_speech_pipeline(
        audio_path=audio_path,
        forced_language=lang,
        skip_emotion=skip_emotion,
    )

    if not segments:
        print("\n⚠️  Pipeline returned 0 segments.")
        print("   Possible causes:")
        print("   1. Audio contains no recognisable speech (synthetic WAV).")
        print("   2. Whisper VAD filtered all audio as silence.")
        print("   → Try: python tests/test_speech_pipeline.py --audio your_real_call.wav\n")
        return

    # ── Print results ──────────────────────────────────────────────────────
    print(f"\n✅  Pipeline complete — {len(segments)} segment(s) returned\n")
    print(f"{'#':<4} {'Speaker':<14} {'Start':>6} {'End':>6} {'Lang':<5} {'Emotion':<12} {'EmConf%':>7}  {'TrConf%':>7}")
    print("─" * 65)

    for i, seg in enumerate(segments):
        emotion_label = seg.emotion or "—"
        emotion_conf  = f"{seg.emotion_confidence * 100:.1f}" if seg.emotion else "   —"
        tr_conf       = f"{seg.transcription_confidence * 100:.1f}"
        print(
            f"{i:<4} {seg.speaker_id:<14} "
            f"{seg.start_time:>6.2f} {seg.end_time:>6.2f} "
            f"{seg.language:<5} {emotion_label:<12} {emotion_conf:>7}  "
            f"{tr_conf:>7}"
        )
        # Print full text indented below the row
        if seg.original_text.strip():
            # Word-wrap at 90 chars
            words = seg.original_text.strip().split()
            line, lines = [], []
            for w in words:
                if sum(len(x) + 1 for x in line) + len(w) > 90:
                    lines.append(" ".join(line))
                    line = [w]
                else:
                    line.append(w)
            if line:
                lines.append(" ".join(line))
            for l in lines:
                print(f"     └─ {l}")
        print()

    print("─" * 65)
    print(f"  Audio language  : {segments[0].audio_language}")
    print(f"  Unique speakers : {len({s.speaker_id for s in segments})}")
    print(f"  Total segments  : {len(segments)}")
    avg_tr = sum(s.transcription_confidence for s in segments) / len(segments)
    print(f"  Avg transcr.    : {avg_tr * 100:.1f}%")
    if not skip_emotion:
        valid_emotions = [s for s in segments if s.emotion and s.emotion_confidence > 0]
        if valid_emotions:
            avg_em = sum(s.emotion_confidence for s in valid_emotions) / len(valid_emotions)
            print(f"  Avg emotion conf: {avg_em * 100:.1f}%")

    # ── Full transcript block ──────────────────────────────────────────────
    print("\n" + "═" * 70)
    print("  FULL TRANSCRIPT")
    print("═" * 70)
    for seg in segments:
        role_hint = "  [AGENT]   " if "00" in seg.speaker_id else "  [CUSTOMER]"
        print(f"{role_hint} {seg.speaker_id} [{seg.start_time:.1f}s–{seg.end_time:.1f}s]")
        print(f"             {seg.original_text.strip()}")
        if seg.emotion:
            print(f"             Emotion: {seg.emotion} ({seg.emotion_confidence*100:.1f}%)")
        print()
    print("═" * 70 + "\n")



# ══════════════════════════════════════════════════════════════════════════
# Entry point
# ══════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="Test the ConvI speech pipeline")
    parser.add_argument("--audio",        type=str,  default=None,  help="Path to a WAV file (real speech recommended)")
    parser.add_argument("--lang",         type=str,  default=None,  help="Force language: en | ml")
    parser.add_argument("--skip-emotion", action="store_true",       help="Skip emotion detection (faster)")
    args = parser.parse_args()

    tmp_file   = None
    audio_dir  = ROOT / "audio"

    if args.audio:
        audio_path = Path(args.audio).resolve()
        if not audio_path.exists():
            print(f"❌ File not found: {audio_path}")
            sys.exit(1)
    else:
        # Auto-discover first WAV in audio/ folder
        wav_files = sorted(audio_dir.glob("*.wav"))
        if wav_files:
            audio_path = wav_files[0]
            print(f"\n[AUDIO] Found: {audio_path.name}")
            if len(wav_files) > 1:
                print(f"        ({len(wav_files)} WAVs in audio/ - using first. Pass --audio to choose a specific one.)")
        else:
            print(f"\n📢  No WAVs found in {audio_dir}")
            print("   Generating a speech-like synthetic WAV for pipeline smoke-test …")
            tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False, prefix="convi_test_")
            tmp.close()
            tmp_file   = Path(tmp.name)
            _generate_speech_like_wav(tmp_file)
            audio_path = tmp_file

    try:
        run_test(
            audio_path=audio_path,
            lang=args.lang,
            skip_emotion=args.skip_emotion,
        )
    finally:
        if tmp_file and tmp_file.exists():
            tmp_file.unlink()
            logger.debug(f"[Test] Cleaned up {tmp_file}")


if __name__ == "__main__":
    main()
