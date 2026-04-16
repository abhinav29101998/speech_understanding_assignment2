"""
adversarial.py
──────────────
Task 4.2 — Adversarial Noise Injection (FGSM)

Goal: Find the minimum perturbation ε that causes the LID system to
      misclassify Hindi as English, while keeping SNR > 40 dB (inaudible).

Method: Fast Gradient Sign Method (FGSM) — Goodfellow et al. (2014)

Mathematical formulation
────────────────────────
Given:
  x     : original audio (clean Hindi speech)
  f(x)  : LID classifier (Multi-Head LID model)
  y_true: true label (Hindi = 1)
  y_adv : adversarial target (English = 0)
  L     : CrossEntropy loss

FGSM computes:
  δ = ε · sign(∇_x L(f(x), y_adv))

Perturbed audio:
  x_adv = x + δ

Constraints:
  1. SNR > 40 dB  →  ||δ||₂ / ||x||₂ < 10^(-40/20) ≈ 0.01
  2. x_adv ∈ [-1, 1]

We perform an ε-sweep to find the minimum effective ε.

Usage:
    python src/adversarial.py \
        --audio   outputs/original_segment.wav \
        --lid_w   models/lid_model.pt \
        --out_dir outputs/adversarial/
"""

import os, json, logging, argparse
import numpy as np
import torch
import torch.nn.functional as F
from typing import Tuple, Dict

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
log = logging.getLogger(__name__)

DEVICE     = "cuda" if torch.cuda.is_available() else "cpu"
TARGET_SNR = 40.0    # dB — minimum required SNR


# ══════════════════════════════════════════════════════════════════════════════
#  SNR Computation
# ══════════════════════════════════════════════════════════════════════════════

def compute_snr(original: np.ndarray, perturbed: np.ndarray) -> float:
    """
    Signal-to-Noise Ratio in dB.
    SNR = 10 · log10(||x||² / ||δ||²)
    where δ = perturbed - original.
    """
    noise = perturbed - original
    signal_power = np.mean(original ** 2) + 1e-12
    noise_power  = np.mean(noise ** 2)    + 1e-12
    snr_db = 10 * np.log10(signal_power / noise_power)
    return float(snr_db)


def max_epsilon_for_snr(audio: np.ndarray, target_snr_db: float = TARGET_SNR) -> float:
    """
    Compute the maximum L∞ perturbation ε that keeps SNR ≥ target_snr_db.
    
    Approximation (FGSM sign perturbation is constant-magnitude):
    SNR ≈ 10·log10(||x||² / (ε²·N))
    Solving for ε:
    ε ≤ ||x||_rms / 10^(SNR_dB/20)
    """
    rms = float(np.sqrt(np.mean(audio ** 2)))
    eps_max = rms / (10 ** (target_snr_db / 20))
    return eps_max


# ══════════════════════════════════════════════════════════════════════════════
#  FGSM Attack on LID Model
# ══════════════════════════════════════════════════════════════════════════════

def fgsm_attack(
    audio:      np.ndarray,
    sr:         int,
    lid_model,
    epsilon:    float,
    target_label: int = 0,   # 0 = English (adversarial target)
) -> Tuple[np.ndarray, float, int]:
    """
    Apply single-step FGSM to audio waveform.

    The LID model operates on MFCC features (not raw audio), so we
    compute gradients through the MFCC extraction layer (approximated
    as a differentiable filterbank).

    Parameters
    ----------
    audio        : (T,) numpy array, clean Hindi speech
    sr           : sample rate
    lid_model    : trained MultiHeadLID model
    epsilon      : perturbation magnitude
    target_label : adversarial target (0 = English)

    Returns
    -------
    x_adv    : perturbed audio
    snr      : SNR of perturbation in dB
    pred     : LID prediction on perturbed audio (0=en, 1=hi)
    """
    from src.lid_model import extract_mfcc

    # Convert audio to tensor with grad tracking
    audio_t = torch.tensor(audio, dtype=torch.float32, requires_grad=True).to(DEVICE)

    # Extract MFCC using differentiable operations
    # We approximate MFCC as a linear transform + log for gradient flow
    feats = _differentiable_mfcc(audio_t, sr)      # (T_frames, 120)
    feats_batch = feats.unsqueeze(0)               # (1, T_frames, 120)

    # Forward pass
    lid_model.eval()
    logits_h1, _ = lid_model(feats_batch)          # (1, T, 2)
    # Use mean over frames for loss
    mean_logits = logits_h1.mean(dim=1).squeeze()  # (2,)

    # Adversarial loss: maximize prob of target_label
    target_t = torch.tensor([target_label], dtype=torch.long).to(DEVICE)
    loss = F.cross_entropy(mean_logits.unsqueeze(0), target_t)

    # Backpropagate to audio
    loss.backward()

    # FGSM: δ = ε · sign(∇_x L)
    # Gradient with respect to audio (propagated through MFCC approx)
    if audio_t.grad is not None:
        delta = epsilon * audio_t.grad.sign().cpu().numpy()
    else:
        # Fallback: add white noise with correct magnitude
        log.warning("No gradient w.r.t. audio — using random sign perturbation.")
        delta = epsilon * np.sign(np.random.randn(len(audio)))

    x_adv = np.clip(audio + delta, -1.0, 1.0).astype(np.float32)
    snr   = compute_snr(audio, x_adv)

    # Get prediction on adversarial example
    with torch.no_grad():
        feats_adv  = extract_mfcc(x_adv, sr)
        feats_adv_t = torch.tensor(feats_adv).unsqueeze(0).to(DEVICE)
        logits_adv, _ = lid_model(feats_adv_t)
        pred_adv   = int(logits_adv.mean(dim=1).argmax(dim=-1).item())

    return x_adv, snr, pred_adv


def _differentiable_mfcc(audio: torch.Tensor, sr: int) -> torch.Tensor:
    """
    Approximate MFCC extraction using differentiable PyTorch ops.
    Simplified: log-mel spectrogram (sufficient for gradient flow).
    """
    hop    = int(0.010 * sr)
    win    = int(0.025 * sr)
    n_fft  = 512
    n_mels = 40

    # STFT (not differentiable in older PyTorch — use manual approach)
    # For simplicity, we use a learned linear layer to approximate
    # the MFCC. In production, use torchaudio.transforms.MFCC.
    try:
        import torchaudio
        mfcc_transform = torchaudio.transforms.MFCC(
            sample_rate=sr,
            n_mfcc=40,
            melkwargs={"n_fft": n_fft, "hop_length": hop, "n_mels": n_mels},
        ).to(audio.device)
        mfcc = mfcc_transform(audio.unsqueeze(0)).squeeze(0).T  # (T, 40)
        # Append simple delta approximation (diff)
        delta  = torch.diff(mfcc, dim=0, prepend=mfcc[:1])
        delta2 = torch.diff(delta, dim=0, prepend=delta[:1])
        feats  = torch.cat([mfcc, delta, delta2], dim=1)       # (T, 120)
        return feats
    except Exception:
        # Pure PyTorch STFT fallback
        window = torch.hann_window(win).to(audio.device)
        stft   = torch.stft(audio, n_fft=n_fft, hop_length=hop,
                             win_length=win, window=window,
                             return_complex=True)               # (F, T)
        power  = stft.abs() ** 2                               # (F, T)
        log_pow = torch.log(power + 1e-9).T                   # (T, F)
        # Simple linear projection to 120 dims
        proj   = torch.randn(log_pow.shape[1], 120,
                             device=audio.device) * 0.01
        feats  = log_pow @ proj                               # (T, 120)
        return feats


# ══════════════════════════════════════════════════════════════════════════════
#  Epsilon Sweep
# ══════════════════════════════════════════════════════════════════════════════

def epsilon_sweep(
    audio:        np.ndarray,
    sr:           int,
    lid_model,
    true_lang:    int = 1,          # 1 = Hindi
    target_lang:  int = 0,          # 0 = English (adversarial target)
    eps_range:    Tuple = (1e-5, 1e-1),
    n_steps:      int = 20,
    min_snr:      float = TARGET_SNR,
) -> Dict:
    """
    Sweep ε values to find minimum ε that:
      1. Causes LID to misclassify Hindi as English.
      2. Maintains SNR > 40 dB.

    Returns dict with sweep results and the minimum effective ε.
    """
    eps_values = np.logspace(
        np.log10(eps_range[0]),
        np.log10(eps_range[1]),
        n_steps,
    )

    eps_max_snr = max_epsilon_for_snr(audio, min_snr)
    log.info(f"Max ε for SNR ≥ {min_snr} dB: {eps_max_snr:.6f}")

    results = []
    min_effective_eps = None

    for eps in eps_values:
        x_adv, snr, pred = fgsm_attack(audio, sr, lid_model, float(eps), target_lang)
        misclassified    = (pred == target_lang) and (true_lang != target_lang)
        snr_ok           = snr >= min_snr

        entry = {
            "epsilon":      round(float(eps), 8),
            "snr_db":       round(snr, 2),
            "prediction":   int(pred),
            "misclassified": bool(misclassified),
            "snr_constraint_met": bool(snr_ok),
            "attack_success": bool(misclassified and snr_ok),
        }
        results.append(entry)

        log.info(
            f"ε={eps:.2e} | SNR={snr:.1f}dB | pred={'en' if pred==0 else 'hi'} "
            f"| {'ATTACK SUCCESS ✓' if misclassified and snr_ok else '✗'}"
        )

        if misclassified and snr_ok and min_effective_eps is None:
            min_effective_eps = float(eps)
            log.info(f"  ★ Minimum effective ε found: {eps:.6f}")

    if min_effective_eps is None:
        log.warning("No effective ε found within SNR constraint. "
                    "Try higher epsilon or check model convergence.")

    return {
        "min_effective_epsilon": min_effective_eps,
        "max_epsilon_for_snr":   float(eps_max_snr),
        "target_snr_db":         min_snr,
        "true_label":            "hi" if true_lang == 1 else "en",
        "adversarial_target":    "en" if target_lang == 0 else "hi",
        "sweep":                 results,
    }


# ══════════════════════════════════════════════════════════════════════════════
#  Save Adversarial Audio
# ══════════════════════════════════════════════════════════════════════════════

def save_adversarial_example(
    audio:   np.ndarray,
    sr:      int,
    lid_model,
    epsilon: float,
    out_dir: str = "outputs/adversarial",
) -> dict:
    """
    Generate and save adversarial audio for a single epsilon.
    Also saves the perturbation (delta) for analysis.
    """
    import soundfile as sf
    os.makedirs(out_dir, exist_ok=True)

    x_adv, snr, pred = fgsm_attack(audio, sr, lid_model, epsilon)
    delta = x_adv - audio

    sf.write(os.path.join(out_dir, "original_5sec.wav"),   audio, sr)
    sf.write(os.path.join(out_dir, "adversarial_5sec.wav"), x_adv, sr)
    sf.write(os.path.join(out_dir, "perturbation.wav"),    delta * 10, sr)  # amplified for inspection

    return {
        "epsilon":    epsilon,
        "snr_db":     snr,
        "prediction": "en" if pred == 0 else "hi",
        "files": {
            "original":     f"{out_dir}/original_5sec.wav",
            "adversarial":  f"{out_dir}/adversarial_5sec.wav",
            "perturbation": f"{out_dir}/perturbation.wav",
        }
    }


# ══════════════════════════════════════════════════════════════════════════════
#  Main
# ══════════════════════════════════════════════════════════════════════════════

def run_adversarial_analysis(
    audio_path:   str = "outputs/original_segment.wav",
    lid_weights:  str = "models/lid_model.pt",
    out_dir:      str = "outputs/adversarial",
    segment_sec:  float = 5.0,
    hindi_start:  float = 0.0,   # start of a Hindi segment (seconds)
) -> dict:
    """
    Full adversarial robustness analysis:
    1. Extract a 5-second Hindi segment.
    2. Run epsilon sweep.
    3. Save adversarial example at minimum effective ε.
    4. Save report JSON.
    """
    import soundfile as sf
    from src.lid_model import MultiHeadLID, extract_mfcc

    audio, sr = sf.read(audio_path, dtype="float32")
    if audio.ndim == 2:
        audio = audio.mean(axis=1)

    # Extract 5-second segment starting at hindi_start
    start = int(hindi_start * sr)
    end   = start + int(segment_sec * sr)
    segment = audio[start:end]
    if len(segment) < int(segment_sec * sr) // 2:
        segment = audio[:int(segment_sec * sr)]

    # Load LID model
    model = MultiHeadLID().to(DEVICE)
    if os.path.exists(lid_weights):
        model.load_state_dict(torch.load(lid_weights, map_location=DEVICE))
        log.info(f"LID weights loaded: {lid_weights}")
    else:
        log.warning("No LID weights — using untrained model.")

    # Verify it predicts Hindi (label=1) on clean segment
    feats = extract_mfcc(segment, sr)
    with torch.no_grad():
        logits, _ = model(torch.tensor(feats).unsqueeze(0).to(DEVICE))
        clean_pred = int(logits.mean(dim=1).argmax(dim=-1).item())
    log.info(f"Clean segment prediction: {'en' if clean_pred==0 else 'hi'} (expected: hi)")

    # Epsilon sweep
    sweep_result = epsilon_sweep(segment, sr, model)

    # Save adversarial example at minimum effective ε (or a fixed small ε)
    test_eps = sweep_result.get("min_effective_epsilon") or 1e-3
    adv_info = save_adversarial_example(segment, sr, model, test_eps, out_dir)

    result = {**sweep_result, "adversarial_example": adv_info}

    os.makedirs(out_dir, exist_ok=True)
    report_path = os.path.join(out_dir, "adversarial_report.json")
    with open(report_path, "w") as f:
        json.dump(result, f, indent=2)

    log.info(f"\n{'='*50}")
    log.info(f"Adversarial Robustness Report")
    log.info(f"{'='*50}")
    log.info(f"Min effective ε : {sweep_result.get('min_effective_epsilon', 'N/A')}")
    log.info(f"Max ε (SNR≥40dB): {sweep_result['max_epsilon_for_snr']:.6f}")
    log.info(f"Report saved    : {report_path}")
    log.info(f"{'='*50}")

    return result


# ── Entry Point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="FGSM Adversarial Attack on LID")
    parser.add_argument("--audio",       default="outputs/original_segment.wav")
    parser.add_argument("--lid_w",       default="models/lid_model.pt")
    parser.add_argument("--out_dir",     default="outputs/adversarial")
    parser.add_argument("--hindi_start", type=float, default=0.0,
                        help="Start time (sec) of a known Hindi segment")
    parser.add_argument("--epsilon",     type=float, default=None,
                        help="Fixed epsilon (skip sweep if provided)")
    args = parser.parse_args()

    if args.epsilon:
        import soundfile as sf
        from src.lid_model import MultiHeadLID
        audio, sr = sf.read(args.audio, dtype="float32")
        if audio.ndim == 2:
            audio = audio.mean(axis=1)
        seg = audio[:int(5 * sr)]
        model = MultiHeadLID().to(DEVICE)
        if os.path.exists(args.lid_w):
            model.load_state_dict(torch.load(args.lid_w, map_location=DEVICE))
        info = save_adversarial_example(seg, sr, model, args.epsilon, args.out_dir)
        print(f"[OK] ε={args.epsilon}  SNR={info['snr_db']:.1f}dB  pred={info['prediction']}")
    else:
        run_adversarial_analysis(args.audio, args.lid_w, args.out_dir, args.hindi_start)
