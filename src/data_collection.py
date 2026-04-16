"""
data_collection.py
──────────────────
Task: Download the lecture from YouTube and extract the 10-minute segment
      between timestamps 2h20m – 2h54m (we pick the middle 10 min = 2h32m–2h42m).

Usage:
    python src/data_collection.py
"""

import os
import subprocess
import sys
import argparse
import logging

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
log = logging.getLogger(__name__)

# ── CONFIG ─────────────────────────────────────────────────────────────

YOUTUBE_URL   = "https://youtu.be/ZPUtA3W-7_I?si=wCClM6UD1HmuYHTa"

SEG_START_SEC = 2 * 3600 + 32 * 60
SEG_END_SEC   = 2 * 3600 + 42 * 60
SEG_DURATION  = SEG_END_SEC - SEG_START_SEC  # 600 sec

OUTPUT_DIR    = "outputs"
RAW_AUDIO     = os.path.join(OUTPUT_DIR, "lecture_full.%(ext)s")
SEGMENT_WAV   = os.path.join(OUTPUT_DIR, "original_segment.wav")
SAMPLE_RATE   = 16000

# 🔥 Absolute FFmpeg path (IMPORTANT)
FFMPEG_PATH = r"G:\Downloads\ffmpeg-2026-04-09-git-d3d0b7a5ee-full_build\bin\ffmpeg.exe"

# Add FFmpeg to PATH (extra safety)
os.environ["PATH"] += os.pathsep + os.path.dirname(FFMPEG_PATH)


# ── FUNCTIONS ───────────────────────────────────────────────────────────

def download_audio() -> str:
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    raw_path = os.path.join(OUTPUT_DIR, "lecture_full")

    # Check yt-dlp
    try:
        subprocess.run(["yt-dlp", "--version"], capture_output=True, check=True)
    except:
        log.error("yt-dlp not found. Install with: pip install yt-dlp")
        sys.exit(1)

    # Already downloaded?
    for ext in ["m4a", "webm", "opus", "mp3", "wav"]:
        if os.path.exists(f"{raw_path}.{ext}"):
            log.info(f"Audio already downloaded: {raw_path}.{ext}")
            return f"{raw_path}.{ext}"

    log.info("Downloading audio from YouTube …")

    cmd = [
        "yt-dlp",
        "--format", "bestaudio/best",
        "--extract-audio",
        "--audio-format", "wav",
        "--audio-quality", "0",
        "--output", f"{raw_path}.%(ext)s",
        "--no-playlist",
        YOUTUBE_URL,
    ]

    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode != 0:
        log.error(f"yt-dlp failed:\n{result.stderr}")
        sys.exit(1)

    for ext in ["wav", "m4a", "webm", "opus"]:
        if os.path.exists(f"{raw_path}.{ext}"):
            log.info(f"Downloaded: {raw_path}.{ext}")
            return f"{raw_path}.{ext}"

    log.error("Could not locate downloaded file.")
    sys.exit(1)


def extract_segment(input_path: str, output_path: str = SEGMENT_WAV) -> str:
    # 🔥 Direct file existence check (no subprocess issue)
    if not os.path.exists(FFMPEG_PATH):
        log.error(f"FFmpeg not found at: {FFMPEG_PATH}")
        sys.exit(1)

    print("Using FFmpeg at:", FFMPEG_PATH)

    if os.path.exists(output_path):
        log.info(f"Segment already exists: {output_path}")
        return output_path

    start_ts = _seconds_to_ts(SEG_START_SEC)

    log.info(f"Extracting segment {start_ts} + {SEG_DURATION}s → {output_path} …")

    cmd = [
        FFMPEG_PATH, "-y",
        "-i", input_path,
        "-ss", str(SEG_START_SEC),
        "-t", str(SEG_DURATION),
        "-ar", str(SAMPLE_RATE),
        "-ac", "1",
        "-acodec", "pcm_s16le",
        output_path,
    ]

    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode != 0:
        log.error(f"ffmpeg failed:\n{result.stderr}")
        sys.exit(1)

    size_mb = os.path.getsize(output_path) / 1e6
    log.info(f"Segment saved: {output_path} ({size_mb:.1f} MB)")

    return output_path


def _seconds_to_ts(sec: int) -> str:
    h, rem = divmod(sec, 3600)
    m, s   = divmod(rem, 60)
    return f"{h:02d}:{m:02d}:{s:02d}"


def verify_segment(path: str) -> None:
    try:
        import soundfile as sf
        info = sf.info(path)
        dur  = info.frames / info.samplerate

        log.info(
            f"Verification → sr={info.samplerate} Hz | "
            f"duration={dur:.1f}s | channels={info.channels}"
        )

        assert abs(dur - SEG_DURATION) < 2
        assert info.samplerate == SAMPLE_RATE

        log.info("✓ Segment verified successfully.")

    except ImportError:
        log.warning("soundfile not installed — skipping verification.")


# ── MAIN ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download & extract lecture segment")

    parser.add_argument("--skip-download", action="store_true")
    parser.add_argument("--raw-path", type=str, default=None)

    args = parser.parse_args()

    if args.raw_path:
        raw = args.raw_path
    else:
        raw = download_audio()

    seg = extract_segment(raw)
    verify_segment(seg)

    print(f"\n[OK] Segment ready at: {seg}")