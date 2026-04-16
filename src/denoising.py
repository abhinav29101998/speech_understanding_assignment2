"""
denoising.py
────────────
Task 1.3: Denoising & Normalization

Two denoising strategies implemented:
  1. Spectral Subtraction  — classical, always available (primary fallback)
  2. DeepFilterNet         — deep-learning denoiser (used if installed)

Usage:
    python src/denoising.py --input outputs/original_segment.wav \
                            --output outputs/denoised_segment.wav \
                            --method spectral   # or 'deepfilter'
"""

import os
import argparse
import logging
import numpy as np

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
log = logging.getLogger(__name__)


# ══════════════════════════════════════════════════════════════════════════════
#  Strategy 1 — Spectral Subtraction
# ══════════════════════════════════════════════════════════════════════════════

def spectral_subtraction(
    audio: np.ndarray,
    sr: int,
    frame_len: float = 0.025,   # 25 ms
    frame_hop: float = 0.010,   # 10 ms
    n_fft: int = 512,
    noise_frames: int = 20,     # initial silence frames for noise estimate
    alpha: float = 2.0,         # over-subtraction factor
    beta: float  = 0.002,       # spectral floor
) -> np.ndarray:
    """
    Classic power spectral subtraction (Boll, 1979).

    Y(ω) = max( |X(ω)|² − α·|N(ω)|², β·|X(ω)|² )^0.5  · e^{j·∠X(ω)}

    where N(ω) is the noise PSD estimated from the first `noise_frames` frames.
    """
    import scipy.signal as signal

    hop  = int(frame_hop * sr)
    win  = int(frame_len * sr)

    # STFT
    _, _, S = signal.stft(audio, fs=sr, nperseg=win, noverlap=win - hop,
                          nfft=n_fft, window="hann")
    power = np.abs(S) ** 2

    # Estimate noise PSD from first `noise_frames` frames
    noise_psd = np.mean(power[:, :noise_frames], axis=1, keepdims=True)

    # Subtraction with floor
    enhanced_power = np.maximum(power - alpha * noise_psd, beta * power)
    enhanced_mag   = np.sqrt(enhanced_power)

    # Reconstruct phase from original
    S_enhanced = enhanced_mag * np.exp(1j * np.angle(S))

    # ISTFT
    _, audio_out = signal.istft(S_enhanced, fs=sr, nperseg=win,
                                noverlap=win - hop, nfft=n_fft, window="hann")

    # Match length
    audio_out = audio_out[: len(audio)]
    return audio_out.astype(np.float32)


# ══════════════════════════════════════════════════════════════════════════════
#  Strategy 2 — DeepFilterNet
# ══════════════════════════════════════════════════════════════════════════════

def deepfilter_denoise(audio: np.ndarray, sr: int) -> np.ndarray:
    """
    Use DeepFilterNet (Schröter et al., 2022) for RNNoise-level denoising.
    Falls back to spectral subtraction if the package is unavailable.
    """
    try:
        from df.enhance import enhance, init_df, load_audio, save_audio
        import tempfile, soundfile as sf

        log.info("DeepFilterNet available — using neural denoiser.")
        model, df_state, _ = init_df()

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_in:
            sf.write(tmp_in.name, audio, sr, subtype="PCM_16")
            tmp_in_name = tmp_in.name

        audio_df, sr_df = load_audio(tmp_in_name, sr=df_state.sr())
        enhanced = enhance(model, df_state, audio_df)

        # Resample back if needed
        if sr_df != sr:
            import librosa
            enhanced = librosa.resample(enhanced.numpy(), orig_sr=sr_df, target_sr=sr)
        else:
            enhanced = enhanced.numpy()

        os.unlink(tmp_in_name)
        return enhanced.astype(np.float32)

    except ImportError:
        log.warning("DeepFilterNet not installed → falling back to spectral subtraction.")
        return spectral_subtraction(audio, sr)


# ══════════════════════════════════════════════════════════════════════════════
#  Normalization
# ══════════════════════════════════════════════════════════════════════════════

def normalize_audio(audio: np.ndarray, target_db: float = -23.0) -> np.ndarray:
    """
    Loudness-normalize to target dBFS (ITU-R BS.1770 approximation).
    """
    rms = np.sqrt(np.mean(audio ** 2) + 1e-9)
    rms_db = 20 * np.log10(rms)
    gain   = 10 ** ((target_db - rms_db) / 20)
    audio_norm = audio * gain
    # Hard clip protection
    audio_norm = np.clip(audio_norm, -1.0, 1.0)
    return audio_norm.astype(np.float32)


# ══════════════════════════════════════════════════════════════════════════════
#  Public API
# ══════════════════════════════════════════════════════════════════════════════

def denoise(
    input_path: str,
    output_path: str,
    method: str = "spectral",
) -> str:
    """
    Denoise an audio file and save the result.

    Parameters
    ----------
    input_path  : path to noisy WAV
    output_path : path to write denoised WAV
    method      : 'spectral' | 'deepfilter'

    Returns
    -------
    output_path
    """
    import soundfile as sf

    log.info(f"Loading: {input_path}")
    audio, sr = sf.read(input_path, dtype="float32")

    # Ensure mono
    if audio.ndim == 2:
        audio = audio.mean(axis=1)

    log.info(f"Denoising with method='{method}' …")
    if method == "deepfilter":
        denoised = deepfilter_denoise(audio, sr)
    else:
        denoised = spectral_subtraction(audio, sr)

    denoised = normalize_audio(denoised)

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    sf.write(output_path, denoised, sr, subtype="PCM_16")
    log.info(f"Denoised audio saved: {output_path}")
    return output_path


# ── Entry Point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Denoise lecture audio")
    parser.add_argument("--input",  default="outputs/original_segment.wav")
    parser.add_argument("--output", default="outputs/denoised_segment.wav")
    parser.add_argument("--method", default="spectral",
                        choices=["spectral", "deepfilter"])
    args = parser.parse_args()

    denoise(args.input, args.output, args.method)
    print(f"[OK] Denoised file: {args.output}")
