"""
anti_spoofing.py
────────────────
Task 4.1 — Anti-Spoofing Countermeasure (CM) System

Implements an LFCC (Linear Frequency Cepstral Coefficients) based
binary classifier to detect:
  - Bona Fide : real human speech (student_voice_ref.wav)
  - Spoof     : synthesized TTS output (output_LRL_cloned.wav)

Architecture:
  • Feature  : LFCC (60 coefficients + Δ + ΔΔ = 180-dim)
  • Model    : Light-CNN (LCNN) — standard ASVspoof baseline
              [Conv2d → MaxPool → FC → Sigmoid]
  • Loss     : Weighted Binary Cross-Entropy
  • Eval     : Equal Error Rate (EER) — target < 10%

EER definition:
  EER = threshold t* where FAR(t*) == FRR(t*)
  FAR = False Acceptance Rate (spoof accepted as bona fide)
  FRR = False Rejection Rate (bona fide rejected as spoof)

Usage:
    # Train
    python src/anti_spoofing.py --train \
        --bona  outputs/student_voice_ref.wav \
        --spoof outputs/output_LRL_cloned.wav

    # Evaluate
    python src/anti_spoofing.py --eval \
        --bona  outputs/student_voice_ref.wav \
        --spoof outputs/output_LRL_cloned.wav \
        --weights models/cm_model.pt
"""

import os, json, logging, argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import roc_curve
from scipy.optimize import brentq
from scipy.interpolate import interp1d
from typing import List, Tuple

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
log = logging.getLogger(__name__)

DEVICE   = "cuda" if torch.cuda.is_available() else "cpu"
N_LFCC   = 60
SR       = 16000
FRAME_LEN = 0.025
FRAME_HOP = 0.010


# ══════════════════════════════════════════════════════════════════════════════
#  LFCC Feature Extraction
# ══════════════════════════════════════════════════════════════════════════════

def linear_filterbank(sr: int, n_fft: int, n_filters: int) -> np.ndarray:
    """
    Construct a linear (not mel) filterbank matrix.
    Shape: (n_filters, n_fft // 2 + 1)

    Unlike Mel filterbanks (logarithmic spacing), LFCC uses linearly
    spaced filters — better for detecting artifacts in synthetic speech
    because synthesis artifacts are often spread linearly in frequency.
    """
    low_freq  = 0.0
    high_freq = sr / 2.0
    freqs     = np.linspace(low_freq, high_freq, n_fft // 2 + 1)
    centers   = np.linspace(low_freq, high_freq, n_filters + 2)

    fb = np.zeros((n_filters, n_fft // 2 + 1))
    for i in range(n_filters):
        left   = centers[i]
        center = centers[i + 1]
        right  = centers[i + 2]
        for j, f in enumerate(freqs):
            if left <= f <= center:
                fb[i, j] = (f - left) / (center - left + 1e-9)
            elif center < f <= right:
                fb[i, j] = (right - f) / (right - center + 1e-9)
    return fb.astype(np.float32)


def extract_lfcc(audio: np.ndarray, sr: int = SR,
                 n_lfcc: int = N_LFCC, n_filters: int = 70) -> np.ndarray:
    """
    Extract LFCC + Δ + ΔΔ features.

    Steps:
      1. Pre-emphasis
      2. Framing + Hamming window
      3. FFT power spectrum
      4. Linear filterbank → log
      5. DCT → LFCC coefficients
      6. Append Δ and ΔΔ

    Returns: (T, n_lfcc * 3)
    """
    import scipy.signal as signal
    import scipy.fft as fft

    # 1. Pre-emphasis
    audio = np.append(audio[0], audio[1:] - 0.97 * audio[:-1])

    hop    = int(FRAME_HOP * sr)
    win    = int(FRAME_LEN * sr)
    n_fft  = 512

    # 2. STFT
    freqs, times, S = signal.stft(audio, fs=sr, nperseg=win, noverlap=win - hop,
                                   nfft=n_fft, window="hamming")
    power = np.abs(S) ** 2  # (F, T)

    # 3. Linear filterbank
    fb = linear_filterbank(sr, n_fft, n_filters)  # (n_filters, F)
    filtered = fb @ power                          # (n_filters, T)
    log_filtered = np.log(filtered + 1e-9)

    # 4. DCT (Type-II) to get LFCC
    from scipy.fft import dct
    lfcc = dct(log_filtered, type=2, axis=0, norm="ortho")[:n_lfcc]  # (n_lfcc, T)
    lfcc = lfcc.T  # (T, n_lfcc)

    # 5. Normalize per utterance
    lfcc = (lfcc - lfcc.mean(0)) / (lfcc.std(0) + 1e-9)

    # 6. Delta and Delta-Delta
    def delta(feat, N=2):
        denom = 2 * sum(i ** 2 for i in range(1, N + 1))
        pad   = np.pad(feat, ((N, N), (0, 0)), mode="edge")
        d     = np.zeros_like(feat)
        for n in range(1, N + 1):
            d += n * (pad[N + n: N + n + len(feat)] - pad[N - n: N - n + len(feat)])
        return d / (denom + 1e-9)

    d1 = delta(lfcc)
    d2 = delta(d1)
    feats = np.concatenate([lfcc, d1, d2], axis=1)  # (T, n_lfcc*3)
    return feats.astype(np.float32)


# ══════════════════════════════════════════════════════════════════════════════
#  LCNN Model (Light-CNN for ASV Anti-Spoofing)
# ══════════════════════════════════════════════════════════════════════════════

class MaxFeatureMap(nn.Module):
    """Max Feature Map activation — used in LCNN architecture."""
    def forward(self, x):
        # Split channels in half and take element-wise max
        x1, x2 = x.chunk(2, dim=1)
        return torch.max(x1, x2)


class LCNNAntiSpoof(nn.Module):
    """
    Light-CNN (LCNN) for anti-spoofing.
    Input: (B, 1, T, n_lfcc*3) — spectrogram-like LFCC feature map
    Output: (B, 2) — [bona fide score, spoof score]

    Reference: Wu et al. (2020) "Light CNN for deep face representation"
               adapted for anti-spoofing by Lavrentyeva et al. (2019).
    """

    def __init__(self, input_dim: int = 180):
        super().__init__()
        self.features = nn.Sequential(
            # Block 1
            nn.Conv2d(1, 32, kernel_size=(5, 5), padding=2),
            MaxFeatureMap(),
            nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2)),

            # Block 2
            nn.Conv2d(16, 64, kernel_size=(1, 1), padding=0),
            MaxFeatureMap(),
            nn.BatchNorm2d(32),
            nn.Conv2d(32, 64, kernel_size=(3, 3), padding=1),
            MaxFeatureMap(),
            nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2)),
            nn.Dropout2d(p=0.25),

            # Block 3
            nn.Conv2d(32, 96, kernel_size=(1, 1), padding=0),
            MaxFeatureMap(),
            nn.BatchNorm2d(48),
            nn.Conv2d(48, 96, kernel_size=(3, 3), padding=1),
            MaxFeatureMap(),
            nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2)),
            nn.Dropout2d(p=0.25),
        )

        self.classifier = nn.Sequential(
            nn.Linear(48 * 4 * 22, 160),   # approximate — adjusted by forward pass
            MaxFeatureMap(),
            nn.Linear(80, 2),
        )

        self._input_dim = input_dim
        self._fc_dim    = None  # computed on first forward pass

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, 1, T, D)"""
        h = self.features(x)          # (B, C, T', D')
        h = h.view(h.size(0), -1)     # flatten

        # Lazy initialization of FC layer
        if self._fc_dim is None or h.size(1) != self._fc_dim:
            self._fc_dim = h.size(1)
            self.classifier = nn.Sequential(
                nn.Linear(self._fc_dim, 160),
                MaxFeatureMap(),
                nn.Linear(80, 2),
            ).to(x.device)

        return self.classifier(h)     # (B, 2)


# ══════════════════════════════════════════════════════════════════════════════
#  Dataset
# ══════════════════════════════════════════════════════════════════════════════

def load_audio_chunks(path: str, chunk_sec: float = 3.0) -> List[np.ndarray]:
    """Load audio and split into fixed-length chunks for training."""
    import soundfile as sf
    audio, sr = sf.read(path, dtype="float32")
    if audio.ndim == 2:
        audio = audio.mean(axis=1)
    if sr != SR:
        import librosa
        audio = librosa.resample(audio, orig_sr=sr, target_sr=SR)

    hop    = int(chunk_sec * SR)
    chunks = []
    for i in range(0, len(audio) - hop + 1, hop // 2):   # 50% overlap
        chunks.append(audio[i: i + hop])
    return chunks


class AntiSpoofDataset(Dataset):
    def __init__(self, bona_chunks, spoof_chunks, max_frames=300):
        self.samples   = []
        self.max_frames = max_frames
        for chunk in bona_chunks:
            feats = extract_lfcc(chunk)
            self.samples.append((self._pad(feats), 0))  # label 0 = bona fide
        for chunk in spoof_chunks:
            feats = extract_lfcc(chunk)
            self.samples.append((self._pad(feats), 1))  # label 1 = spoof

    def _pad(self, feats: np.ndarray) -> np.ndarray:
        T, D = feats.shape
        if T >= self.max_frames:
            return feats[:self.max_frames]
        pad = np.zeros((self.max_frames - T, D), dtype=np.float32)
        return np.vstack([feats, pad])

    def __len__(self):  return len(self.samples)

    def __getitem__(self, i):
        feats, label = self.samples[i]
        x = torch.tensor(feats).unsqueeze(0)  # (1, T, D)
        return x, label


# ══════════════════════════════════════════════════════════════════════════════
#  EER Computation
# ══════════════════════════════════════════════════════════════════════════════

def compute_eer(scores: np.ndarray, labels: np.ndarray) -> Tuple[float, float]:
    """
    Compute Equal Error Rate (EER).

    EER = threshold t* where FAR(t*) ≈ FRR(t*)

    Parameters
    ----------
    scores : CM scores (higher = more likely bona fide)
    labels : 0 = bona fide, 1 = spoof

    Returns
    -------
    eer       : EER value (0–1)
    threshold : decision threshold at EER

    Method: argmin |FPR − FNR| over the ROC curve (robust to non-crossing cases).
    Refined with brentq when FPR and FNR strictly cross.
    """
    fpr, tpr, thresholds = roc_curve(labels, -scores, pos_label=1)
    fnr = 1 - tpr

    # Primary: argmin |FPR - FNR|
    diff    = np.abs(fpr - fnr)
    min_idx = int(np.argmin(diff))
    eer       = float((fpr[min_idx] + fnr[min_idx]) / 2.0)
    threshold = float(thresholds[min_idx]) if min_idx < len(thresholds) else 0.5

    # Refinement: brentq when a clean sign-change exists
    try:
        t  = thresholds.astype(float)
        fp = fpr[:len(t)].astype(float)
        fn = fnr[:len(t)].astype(float)
        diff_arr = fp - fn
        # Find consecutive pairs with sign change
        for i in range(len(diff_arr) - 1):
            if diff_arr[i] * diff_arr[i + 1] < 0:
                t_lo, t_hi = float(t[i + 1]), float(t[i])
                if t_lo > t_hi:
                    t_lo, t_hi = t_hi, t_lo
                fp_i = interp1d(t, fp, fill_value="extrapolate")
                fn_i = interp1d(t, fn, fill_value="extrapolate")
                t_star = brentq(lambda x: float(fp_i(x)) - float(fn_i(x)),
                                t_lo, t_hi, maxiter=100)
                eer       = float((fp_i(t_star) + fn_i(t_star)) / 2.0)
                threshold = float(t_star)
                break
    except Exception:
        pass  # keep argmin result

    return eer, threshold


# ══════════════════════════════════════════════════════════════════════════════
#  Training
# ══════════════════════════════════════════════════════════════════════════════

def train(
    bona_path:  str = "outputs/student_voice_ref.wav",
    spoof_path: str = "outputs/output_LRL_cloned.wav",
    save_path:  str = "models/cm_model.pt",
    epochs:     int = 40,
    lr:         float = 1e-3,
):
    log.info("Loading audio for anti-spoofing training …")
    bona_chunks  = load_audio_chunks(bona_path)
    spoof_chunks = load_audio_chunks(spoof_path)

    log.info(f"Bona fide chunks: {len(bona_chunks)}  |  Spoof chunks: {len(spoof_chunks)}")

    dataset = AntiSpoofDataset(bona_chunks, spoof_chunks)
    split   = int(0.8 * len(dataset))
    train_ds, val_ds = torch.utils.data.random_split(
        dataset, [split, len(dataset) - split]
    )
    train_dl = DataLoader(train_ds, batch_size=16, shuffle=True)
    val_dl   = DataLoader(val_ds,   batch_size=16, shuffle=False)

    model = LCNNAntiSpoof().to(DEVICE)
    # Compute class weights (handle imbalance)
    n_bona  = len(bona_chunks)
    n_spoof = len(spoof_chunks)
    w = torch.tensor(
        [n_spoof / (n_bona + n_spoof), n_bona / (n_bona + n_spoof)],
        dtype=torch.float32
    ).to(DEVICE)
    criterion = nn.CrossEntropyLoss(weight=w)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)

    best_eer = 1.0
    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        for x, y in train_dl:
            x, y = x.to(DEVICE), y.to(DEVICE)
            logits = model(x)
            loss   = criterion(logits, y)
            optimizer.zero_grad(); loss.backward(); optimizer.step()
            total_loss += loss.item()
        scheduler.step()

        # Validation EER
        model.eval()
        all_scores, all_labels = [], []
        with torch.no_grad():
            for x, y in val_dl:
                logits = model(x.to(DEVICE))
                scores = F.softmax(logits, dim=1)[:, 0].cpu().numpy()  # bona fide score
                all_scores.extend(scores)
                all_labels.extend(y.numpy())

        if len(set(all_labels)) < 2:
            log.warning("Only one class in validation — skipping EER.")
            continue

        eer, thresh = compute_eer(
            np.array(all_scores), np.array(all_labels)
        )
        log.info(f"Epoch {epoch:3d}/{epochs} | loss={total_loss/len(train_dl):.4f} "
                 f"| EER={eer:.4f}  (target <0.10)")

        if eer < best_eer:
            best_eer = eer
            os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
            torch.save(model.state_dict(), save_path)
            log.info(f"  ✓ Best model saved (EER={eer:.4f})")

    log.info(f"Training complete. Best EER = {best_eer:.4f}")
    return best_eer


# ══════════════════════════════════════════════════════════════════════════════
#  Evaluation
# ══════════════════════════════════════════════════════════════════════════════

def evaluate(
    bona_path:    str = "outputs/student_voice_ref.wav",
    spoof_path:   str = "outputs/output_LRL_cloned.wav",
    weights_path: str = "models/cm_model.pt",
) -> dict:
    """
    Evaluate the CM system on bona fide vs spoof audio.
    Prints EER and returns evaluation dict.
    """
    import soundfile as sf

    model = LCNNAntiSpoof().to(DEVICE)
    if os.path.exists(weights_path):
        model.load_state_dict(torch.load(weights_path, map_location=DEVICE))
        log.info(f"Loaded CM weights: {weights_path}")
    else:
        log.warning("No trained weights found — using untrained model.")

    model.eval()

    bona_chunks  = load_audio_chunks(bona_path)
    spoof_chunks = load_audio_chunks(spoof_path)
    dataset      = AntiSpoofDataset(bona_chunks, spoof_chunks)
    dl           = DataLoader(dataset, batch_size=16, shuffle=False)

    all_scores, all_labels = [], []
    with torch.no_grad():
        for x, y in dl:
            logits = model(x.to(DEVICE))
            scores = F.softmax(logits, dim=1)[:, 0].cpu().numpy()
            all_scores.extend(scores)
            all_labels.extend(y.numpy())

    scores = np.array(all_scores)
    labels = np.array(all_labels)

    eer, threshold = compute_eer(scores, labels)

    # Confusion matrix at EER threshold
    preds = (scores < threshold).astype(int)  # <threshold → spoof
    TP = int(np.sum((preds == 1) & (labels == 1)))
    TN = int(np.sum((preds == 0) & (labels == 0)))
    FP = int(np.sum((preds == 1) & (labels == 0)))
    FN = int(np.sum((preds == 0) & (labels == 1)))

    result = {
        "EER":              round(eer, 4),
        "threshold":        round(threshold, 4),
        "TP": TP, "TN": TN, "FP": FP, "FN": FN,
        "accuracy":         round((TP + TN) / max(TP + TN + FP + FN, 1), 4),
        "target_met":       eer < 0.10,
    }

    log.info(f"\n{'='*50}")
    log.info(f"Anti-Spoofing Evaluation Results")
    log.info(f"{'='*50}")
    log.info(f"EER       : {eer:.4f}  ({'✓ PASS' if eer < 0.10 else '✗ FAIL'} — target <10%)")
    log.info(f"Threshold : {threshold:.4f}")
    log.info(f"Accuracy  : {result['accuracy']:.4f}")
    log.info(f"Confusion : TP={TP} TN={TN} FP={FP} FN={FN}")
    log.info(f"{'='*50}")

    return result


# ── Entry Point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Anti-Spoofing CM (LFCC + LCNN)")
    parser.add_argument("--train",   action="store_true")
    parser.add_argument("--eval",    action="store_true")
    parser.add_argument("--bona",    default="outputs/student_voice_ref.wav")
    parser.add_argument("--spoof",   default="outputs/output_LRL_cloned.wav")
    parser.add_argument("--weights", default="models/cm_model.pt")
    parser.add_argument("--epochs",  type=int, default=40)
    parser.add_argument("--out",     default="outputs/cm_eval.json")
    args = parser.parse_args()

    if args.train:
        train(args.bona, args.spoof, args.weights, epochs=args.epochs)
    if args.eval:
        result = evaluate(args.bona, args.spoof, args.weights)
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        with open(args.out, "w") as f:
            json.dump(result, f, indent=2)
        print(f"\n[OK] CM evaluation saved: {args.out}")
        print(f"     EER = {result['EER']:.4f}  |  Target: <0.10  |  "
              f"{'PASS ✓' if result['target_met'] else 'FAIL ✗'}")
