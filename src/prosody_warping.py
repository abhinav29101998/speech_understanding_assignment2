"""
prosody_warping.py
──────────────────
Task 3.2 — Prosody Warping via Dynamic Time Warping (DTW)

Extracts Fundamental Frequency (F₀) and Energy contours from the
professor's lecture audio, then applies DTW to warp these prosodic
features onto the synthesized LRL speech, preserving "teaching style".

Mathematical foundation
───────────────────────
Given source contour X = [x₁, …, xₙ] and target contour Y = [y₁, …, yₘ],
DTW finds the optimal monotone alignment path π* through the cost matrix C:

    C(i, j) = |x_i − y_j|

    D(i, j) = C(i, j) + min(D(i−1, j), D(i, j−1), D(i−1, j−1))

The warping function φ: [1…n] → [1…m] is extracted from the backtrace of D.
We then resample the source prosody using φ and apply it to the target audio.

F₀ extraction: Praat via parselmouth (autocorrelation + SHR).
Energy: RMS per frame.
Warping application: PSOLA-based pitch modification via librosa / pyworld.

Usage:
    python src/prosody_warping.py \
        --source outputs/original_segment.wav \
        --target outputs/tts_raw.wav \
        --out    outputs/tts_warped.wav
"""

import os, logging, argparse
import numpy as np
from typing import Tuple, Optional

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
log = logging.getLogger(__name__)

FRAME_HOP = 0.010   # 10 ms hop for prosody analysis
FRAME_LEN = 0.025   # 25 ms window


# ══════════════════════════════════════════════════════════════════════════════
#  F₀ Extraction
# ══════════════════════════════════════════════════════════════════════════════

def extract_f0_parselmouth(audio: np.ndarray, sr: int) -> np.ndarray:
    """
    Extract F₀ contour using Praat via parselmouth.
    Unvoiced frames are set to 0.0.
    Returns f0 array of shape (T,) where T = num frames.
    """
    import parselmouth
    from parselmouth.praat import call

    snd    = parselmouth.Sound(audio, sampling_frequency=sr)
    pitch  = call(snd, "To Pitch", FRAME_HOP, 75.0, 600.0)  # 75–600 Hz range
    t_step = pitch.dt
    f0_arr = np.array([pitch.get_value_at_time(t)
                       for t in np.arange(0, pitch.duration, t_step)])
    f0_arr = np.nan_to_num(f0_arr, nan=0.0)
    log.info(f"F₀ extracted: {len(f0_arr)} frames  "
             f"(voiced={np.sum(f0_arr > 0)} / {len(f0_arr)})")
    return f0_arr.astype(np.float32)


def extract_f0_librosa(audio: np.ndarray, sr: int) -> np.ndarray:
    """Fallback F₀ extraction using librosa pyin."""
    import librosa
    hop = int(FRAME_HOP * sr)
    f0, voiced_flag, _ = librosa.pyin(
        audio, fmin=75, fmax=600,
        sr=sr, hop_length=hop,
        fill_na=0.0,
    )
    f0 = np.nan_to_num(f0, nan=0.0)
    log.info(f"F₀ (pyin) extracted: {len(f0)} frames")
    return f0.astype(np.float32)


def extract_f0(audio: np.ndarray, sr: int) -> np.ndarray:
    try:
        return extract_f0_parselmouth(audio, sr)
    except Exception as e:
        log.warning(f"Parselmouth F₀ failed ({e}) — using librosa pyin.")
        return extract_f0_librosa(audio, sr)


# ══════════════════════════════════════════════════════════════════════════════
#  Energy Contour
# ══════════════════════════════════════════════════════════════════════════════

def extract_energy(audio: np.ndarray, sr: int) -> np.ndarray:
    """
    Compute RMS energy per frame.
    Returns energy array of shape (T,).
    """
    hop  = int(FRAME_HOP * sr)
    win  = int(FRAME_LEN * sr)
    n_frames = (len(audio) - win) // hop + 1
    energy = np.array([
        np.sqrt(np.mean(audio[i * hop: i * hop + win] ** 2 + 1e-10))
        for i in range(n_frames)
    ])
    return energy.astype(np.float32)


# ══════════════════════════════════════════════════════════════════════════════
#  Dynamic Time Warping
# ══════════════════════════════════════════════════════════════════════════════

def dtw(x: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray, float]:
    """
    Standard DTW between sequences x (length n) and y (length m).

    Returns
    -------
    path_x   : indices in x along optimal path
    path_y   : indices in y along optimal path
    distance : total DTW distance
    """
    n, m = len(x), len(y)
    # Cost matrix
    D = np.full((n + 1, m + 1), np.inf)
    D[0, 0] = 0.0

    for i in range(1, n + 1):
        for j in range(1, m + 1):
            cost    = abs(float(x[i - 1]) - float(y[j - 1]))
            D[i, j] = cost + min(D[i - 1, j], D[i, j - 1], D[i - 1, j - 1])

    # Backtrace
    path_x, path_y = [], []
    i, j = n, m
    while i > 0 or j > 0:
        path_x.append(i - 1)
        path_y.append(j - 1)
        if i == 0:
            j -= 1
        elif j == 0:
            i -= 1
        else:
            move = np.argmin([D[i - 1, j], D[i, j - 1], D[i - 1, j - 1]])
            if move == 0:   i -= 1
            elif move == 1: j -= 1
            else:           i -= 1; j -= 1

    path_x = np.array(path_x[::-1])
    path_y = np.array(path_y[::-1])
    return path_x, path_y, D[n, m]


def warp_contour(source: np.ndarray, path_src: np.ndarray,
                  path_tgt: np.ndarray, target_len: int) -> np.ndarray:
    """
    Warp source contour to match target length using DTW path.

    We build a mapping from each target frame to a source frame,
    then interpolate the source contour at those positions.
    """
    # Build target→source frame mapping by inverting path
    tgt_to_src = np.interp(
        np.arange(target_len),
        path_tgt,
        path_src.astype(float),
    )
    tgt_to_src = np.clip(tgt_to_src, 0, len(source) - 1).astype(int)
    warped = source[tgt_to_src]
    return warped


# ══════════════════════════════════════════════════════════════════════════════
#  Apply Warped Prosody to Audio (PSOLA-like via pyworld)
# ══════════════════════════════════════════════════════════════════════════════

def apply_f0_and_energy(
    audio: np.ndarray,
    sr: int,
    f0_target: np.ndarray,
    energy_target: np.ndarray,
) -> np.ndarray:
    """
    Apply a target F₀ contour and energy envelope to audio using pyworld.

    pyworld (WORLD vocoder):
      1. Analysis  : DIO (F0) + CheapTrick (SP) + D4C (AP)
      2. Modify    : replace F0 with f0_target
      3. Synthesis : WORLD synthesis

    Falls back to simple pitch-shift + gain if pyworld unavailable.
    """
    try:
        import pyworld as pw

        audio_d = audio.astype(np.float64)
        hop_ms  = FRAME_HOP * 1000

        # Analysis
        _f0, t   = pw.dio(audio_d, sr, frame_period=hop_ms)
        f0_pw    = pw.stonemask(audio_d, _f0, t, sr)
        sp       = pw.cheaptrick(audio_d, f0_pw, t, sr)
        ap       = pw.d4c(audio_d, f0_pw, t, sr)

        # Replace F₀ with warped professor's F₀
        n_frames_pw = len(f0_pw)
        f0_interp   = np.interp(
            np.linspace(0, 1, n_frames_pw),
            np.linspace(0, 1, len(f0_target)),
            f0_target,
        )
        # Keep voicing decisions from original (voiced=nonzero F0)
        voiced_mask  = f0_pw > 0
        f0_modified  = f0_interp * voiced_mask

        # Synthesis
        audio_out = pw.synthesize(f0_modified, sp, ap, sr,
                                   frame_period=hop_ms)

        # Apply energy envelope (simple gain)
        energy_orig    = extract_energy(audio, sr)
        energy_tgt_seg = np.interp(
            np.linspace(0, 1, len(energy_orig)),
            np.linspace(0, 1, len(energy_target)),
            energy_target,
        )
        gain_per_frame = energy_tgt_seg / (energy_orig + 1e-9)
        hop_samples    = int(FRAME_HOP * sr)
        for i in range(len(energy_orig)):
            start = i * hop_samples
            end   = start + hop_samples
            audio_out[start:end] *= np.clip(gain_per_frame[i], 0.1, 10.0)

        audio_out = np.clip(audio_out, -1.0, 1.0)
        log.info("Prosody warping applied via WORLD vocoder.")
        return audio_out.astype(np.float32)

    except ImportError:
        log.warning("pyworld not installed — using simple gain-based warping.")
        # Simple fallback: just apply energy scaling
        energy_orig = extract_energy(audio, sr)
        energy_tgt  = np.interp(
            np.linspace(0, 1, len(energy_orig)),
            np.linspace(0, 1, len(energy_target)),
            energy_target,
        )
        gain = energy_tgt / (energy_orig + 1e-9)
        hop_samples = int(FRAME_HOP * sr)
        audio_out = audio.copy()
        for i in range(len(energy_orig)):
            s = i * hop_samples
            e = min(s + hop_samples, len(audio_out))
            audio_out[s:e] *= np.clip(float(gain[i]), 0.1, 5.0)
        return np.clip(audio_out, -1.0, 1.0).astype(np.float32)


# ══════════════════════════════════════════════════════════════════════════════
#  Main Warping Pipeline
# ══════════════════════════════════════════════════════════════════════════════

def warp_prosody(
    source_path: str,          # professor's lecture (for prosody extraction)
    target_path: str,          # raw TTS output (to be warped)
    out_path:    str,          # warped audio output
    save_plots:  bool = True,
) -> str:
    """
    Full prosody warping pipeline.

    1. Extract F₀ and Energy from source (professor lecture).
    2. Extract F₀ and Energy from target (TTS output).
    3. DTW-align source prosody to target length.
    4. Apply warped prosody to target audio via WORLD vocoder.
    """
    import soundfile as sf

    log.info(f"Source (professor): {source_path}")
    log.info(f"Target (TTS raw):   {target_path}")

    src_audio, src_sr = sf.read(source_path, dtype="float32")
    tgt_audio, tgt_sr = sf.read(target_path, dtype="float32")
    if src_audio.ndim == 2: src_audio = src_audio.mean(axis=1)
    if tgt_audio.ndim == 2: tgt_audio = tgt_audio.mean(axis=1)

    # ── Extract prosody ───────────────────────────────────────────────────────
    log.info("Extracting source F₀ and Energy …")
    src_f0     = extract_f0(src_audio, src_sr)
    src_energy = extract_energy(src_audio, src_sr)

    log.info("Extracting target F₀ and Energy …")
    tgt_f0     = extract_f0(tgt_audio, tgt_sr)
    tgt_energy = extract_energy(tgt_audio, tgt_sr)

    # ── DTW alignment ─────────────────────────────────────────────────────────
    log.info("Running DTW on F₀ contours …")
    # Use only voiced frames for DTW (voiced = f0 > 0)
    src_voiced = np.where(src_f0 > 0, src_f0, src_f0.mean())
    tgt_voiced = np.where(tgt_f0 > 0, tgt_f0, tgt_f0.mean())

    path_src, path_tgt, dtw_dist = dtw(src_voiced, tgt_voiced)
    log.info(f"DTW distance: {dtw_dist:.2f}")

    # ── Warp source prosody to target length ──────────────────────────────────
    f0_warped     = warp_contour(src_f0,     path_src, path_tgt, len(tgt_f0))
    energy_warped = warp_contour(src_energy, path_src, path_tgt, len(tgt_energy))

    # ── Apply warped prosody to TTS audio ─────────────────────────────────────
    log.info("Applying warped prosody to TTS audio …")
    warped_audio = apply_f0_and_energy(tgt_audio, tgt_sr, f0_warped, energy_warped)

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    sf.write(out_path, warped_audio, tgt_sr, subtype="PCM_16")
    log.info(f"[OK] Warped audio saved: {out_path}")

    # ── Optional: save prosody plots for report ───────────────────────────────
    if save_plots:
        _save_prosody_plots(src_f0, tgt_f0, f0_warped, src_energy, energy_warped)

    return out_path


def _save_prosody_plots(src_f0, tgt_f0, f0_warped, src_energy, energy_warped):
    try:
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(2, 2, figsize=(14, 8))
        axes[0, 0].plot(src_f0);    axes[0, 0].set_title("Source F₀ (Professor)")
        axes[0, 1].plot(tgt_f0);    axes[0, 1].set_title("Target F₀ (TTS raw)")
        axes[1, 0].plot(f0_warped); axes[1, 0].set_title("Warped F₀ (DTW-aligned)")
        axes[1, 1].plot(src_energy, label="Src", alpha=0.7)
        axes[1, 1].plot(energy_warped, label="Warped", alpha=0.7)
        axes[1, 1].legend(); axes[1, 1].set_title("Energy Contours")
        for ax in axes.flat:
            ax.set_xlabel("Frame"); ax.set_ylabel("Hz / RMS")
        plt.tight_layout()
        os.makedirs("report_assets", exist_ok=True)
        path = "report_assets/prosody_warping.png"
        plt.savefig(path, dpi=150)
        plt.close()
        log.info(f"Prosody plot saved: {path}")
    except Exception as e:
        log.warning(f"Could not save prosody plot: {e}")


# ── Entry Point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="DTW Prosody Warping")
    parser.add_argument("--source", default="outputs/original_segment.wav")
    parser.add_argument("--target", default="outputs/tts_raw.wav")
    parser.add_argument("--out",    default="outputs/tts_warped.wav")
    parser.add_argument("--no-plot", action="store_true")
    args = parser.parse_args()
    warp_prosody(args.source, args.target, args.out,
                 save_plots=not args.no_plot)
