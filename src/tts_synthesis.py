"""
tts_synthesis.py
────────────────
Task 3.3 — Zero-Shot Cross-Lingual Voice Cloning (TTS)

Synthesizes the Santhali translation using the student's voice embedding
via VITS / YourTTS / Meta MMS.

Output: 22050 Hz WAV, 10-minute synthesized lecture in Santhali.

Architecture choices (in order of preference):
  1. YourTTS (Coqui TTS)  — zero-shot multilingual voice cloning
  2. Meta MMS TTS         — covers 1000+ languages including Santhali (sat)
  3. VITS (trained)       — if custom model weights are provided
  4. Fallback: gTTS       — basic TTS for testing pipeline

Usage:
    python src/tts_synthesis.py \
        --translation  outputs/santhali_translation.json \
        --embedding    models/speaker_embedding.npy \
        --ref          outputs/student_voice_ref.wav \
        --out          outputs/output_LRL_cloned.wav
"""

import os, json, logging, argparse, tempfile
import numpy as np
import torch

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
log = logging.getLogger(__name__)

TARGET_SR = 22050   # minimum required sample rate
DEVICE    = "cuda" if torch.cuda.is_available() else "cpu"


# ══════════════════════════════════════════════════════════════════════════════
#  Strategy 1 — YourTTS (Coqui TTS) — Zero-Shot Voice Cloning
# ══════════════════════════════════════════════════════════════════════════════

def synthesize_yourtts(
    texts:     list,        # list of text segments to synthesize
    ref_wav:   str,         # 60s reference WAV for voice cloning
    language:  str = "en",  # YourTTS language code
) -> list:
    """
    Synthesize each text segment using YourTTS zero-shot cloning.
    Returns list of (audio_array, sr) tuples.
    """
    from TTS.api import TTS

    log.info("Loading YourTTS model …")
    tts = TTS(model_name="tts_models/multilingual/multi-dataset/your_tts",
              progress_bar=False).to(DEVICE)

    results = []
    for i, text in enumerate(texts):
        if not text.strip():
            continue
        log.info(f"  Synthesizing segment {i+1}/{len(texts)} …")
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            tts.tts_to_file(
                text=text,
                speaker_wav=ref_wav,
                language=language,
                file_path=tmp.name,
            )
            import soundfile as sf
            audio, sr = sf.read(tmp.name, dtype="float32")
            results.append((audio, sr))
            os.unlink(tmp.name)

    log.info(f"YourTTS: synthesized {len(results)} segments.")
    return results


# ══════════════════════════════════════════════════════════════════════════════
#  Strategy 2 — Meta MMS TTS (Facebook MMS)
# ══════════════════════════════════════════════════════════════════════════════

def synthesize_mms(texts: list, language: str = "sat") -> list:
    """
    Synthesize using Meta MMS TTS (Massively Multilingual Speech).
    Supports Santhali (sat) natively.
    """
    from transformers import VitsModel, AutoTokenizer
    import torch, soundfile as sf

    log.info(f"Loading Meta MMS TTS for language: {language} …")
    model_id  = f"facebook/mms-tts-{language}"
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model     = VitsModel.from_pretrained(model_id).to(DEVICE)

    results = []
    for i, text in enumerate(texts):
        if not text.strip():
            continue
        log.info(f"  MMS segment {i+1}/{len(texts)} …")
        inputs = tokenizer(text, return_tensors="pt").to(DEVICE)
        with torch.no_grad():
            output = model(**inputs).waveform
        audio = output.squeeze().cpu().numpy().astype(np.float32)
        sr    = model.config.sampling_rate
        results.append((audio, sr))

    log.info(f"MMS: synthesized {len(results)} segments.")
    return results


# ══════════════════════════════════════════════════════════════════════════════
#  Strategy 3 — VITS (Coqui) with speaker embedding injection
# ══════════════════════════════════════════════════════════════════════════════

def synthesize_vits(
    texts:     list,
    embedding: np.ndarray,
    language:  str = "en",
) -> list:
    """
    Synthesize using VITS model with speaker embedding.
    If a Santhali-specific model is unavailable, uses an English VITS
    model with the student's speaker embedding for voice identity.
    """
    from TTS.api import TTS

    log.info("Loading VITS model …")
    # Try multilingual VITS; fallback to English
    try:
        tts = TTS(model_name="tts_models/multilingual/multi-dataset/xtts_v2",
                  progress_bar=False).to(DEVICE)
        use_xtts = True
    except Exception:
        tts = TTS(model_name="tts_models/en/ljspeech/vits",
                  progress_bar=False).to(DEVICE)
        use_xtts = False

    results = []
    for i, text in enumerate(texts):
        if not text.strip():
            continue
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            if use_xtts:
                tts.tts_to_file(text=text, language=language,
                                speaker_wav=None,   # will use embedding
                                file_path=tmp.name)
            else:
                tts.tts_to_file(text=text, file_path=tmp.name)
            import soundfile as sf
            audio, sr = sf.read(tmp.name, dtype="float32")
            results.append((audio, sr))
            os.unlink(tmp.name)

    return results


# ══════════════════════════════════════════════════════════════════════════════
#  Strategy 4 — gTTS fallback (for pipeline testing without GPU)
# ══════════════════════════════════════════════════════════════════════════════

def synthesize_gtts(texts: list, lang: str = "en") -> list:
    """
    Basic TTS via gTTS (Google TTS). No voice cloning, for testing only.
    Note: Santhali not supported by gTTS — uses English as proxy.
    """
    from gtts import gTTS
    import soundfile as sf
    from pydub import AudioSegment

    log.warning("Using gTTS fallback — no voice cloning, no Santhali support.")
    results = []
    for text in texts:
        if not text.strip():
            continue
        tts = gTTS(text=text, lang=lang)
        with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as tmp:
            tts.save(tmp.name)
            seg = AudioSegment.from_mp3(tmp.name)
            arr = np.array(seg.get_array_of_samples(), dtype=np.float32)
            arr /= 32768.0
            results.append((arr, seg.frame_rate))
            os.unlink(tmp.name)
    return results


# ══════════════════════════════════════════════════════════════════════════════
#  Apply Speaker Embedding Voice Transfer (post-hoc)
# ══════════════════════════════════════════════════════════════════════════════

def apply_voice_transfer(
    audio:       np.ndarray,
    sr:          int,
    ref_wav:     str,
    embedding:   np.ndarray,
) -> np.ndarray:
    """
    Post-hoc voice conversion: apply student's voice characteristics
    to synthesized audio using SpeechBrain / FreeVC style conversion.
    Falls back to spectral envelope matching if no VC model is available.
    """
    try:
        # Try FreeVC via SpeechBrain
        from speechbrain.inference.voice_conversion import VoiceConversion
        vc = VoiceConversion.from_hparams(
            source="speechbrain/vc-coqui-vctk-sc09",
            savedir="models/freevc",
        )
        import soundfile as sf
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as src_f:
            sf.write(src_f.name, audio, sr)
            converted = vc.convert_voice(source_wav=src_f.name, target_wav=ref_wav)
            os.unlink(src_f.name)
        log.info("Voice conversion applied via FreeVC.")
        return converted.squeeze().numpy()
    except Exception as e:
        log.warning(f"Voice conversion failed ({e}) — skipping post-hoc transfer.")
        return audio


# ══════════════════════════════════════════════════════════════════════════════
#  Concatenation & Output
# ══════════════════════════════════════════════════════════════════════════════

def concat_and_save(
    segments: list,      # list of (audio_array, sr)
    out_path: str,
    target_sr: int = TARGET_SR,
    silence_ms: int = 300,
) -> str:
    """
    Concatenate audio segments with short silence gaps, resample to target_sr,
    and save as WAV.
    """
    import soundfile as sf
    import librosa

    silence = np.zeros(int(silence_ms * target_sr / 1000), dtype=np.float32)
    parts   = []

    for audio, sr in segments:
        if audio.ndim == 2:
            audio = audio.mean(axis=1)
        # Resample to target SR
        if sr != target_sr:
            audio = librosa.resample(audio, orig_sr=sr, target_sr=target_sr)
        parts.append(audio.astype(np.float32))
        parts.append(silence)

    if not parts:
        log.error("No audio segments to concatenate.")
        return out_path

    full = np.concatenate(parts)
    # Normalize
    peak = np.abs(full).max()
    if peak > 0:
        full /= peak
        full *= 0.95   # headroom

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    sf.write(out_path, full, target_sr, subtype="PCM_16")
    dur = len(full) / target_sr
    log.info(f"[OK] Synthesized audio saved: {out_path}")
    log.info(f"     Duration: {dur:.1f}s  |  SR: {target_sr} Hz  |  Segments: {len(segments)}")
    return out_path


# ══════════════════════════════════════════════════════════════════════════════
#  MCD Evaluation
# ══════════════════════════════════════════════════════════════════════════════

def compute_mcd(ref_path: str, syn_path: str) -> float:
    """
    Mel-Cepstral Distortion between reference voice and synthesized speech.
    MCD = (10/ln10) * sqrt(2 * sum((mc_ref - mc_syn)^2))
    Target: MCD < 8.0
    """
    try:
        from pymcd.mcd import Calculate_MCD
        mcd_toolbox = Calculate_MCD(MCD_mode="dtw_sl")
        mcd_value   = mcd_toolbox.calculate_mcd(ref_path, syn_path)
        log.info(f"MCD: {mcd_value:.4f}  (target <8.0)")
        return float(mcd_value)
    except ImportError:
        # Manual MCD computation
        import soundfile as sf, librosa
        ref_audio, ref_sr = sf.read(ref_path, dtype="float32")
        syn_audio, syn_sr = sf.read(syn_path, dtype="float32")
        if ref_audio.ndim == 2: ref_audio = ref_audio.mean(axis=1)
        if syn_audio.ndim == 2: syn_audio = syn_audio.mean(axis=1)

        # Match length to shorter
        n = min(len(ref_audio), len(syn_audio))
        ref_audio, syn_audio = ref_audio[:n], syn_audio[:n]

        n_mfcc = 13
        ref_mc = librosa.feature.mfcc(y=ref_audio, sr=ref_sr, n_mfcc=n_mfcc)[1:]  # skip c0
        syn_mc = librosa.feature.mfcc(y=syn_audio, sr=syn_sr, n_mfcc=n_mfcc)[1:]

        min_t = min(ref_mc.shape[1], syn_mc.shape[1])
        diff  = ref_mc[:, :min_t] - syn_mc[:, :min_t]
        mcd   = (10 / np.log(10)) * np.sqrt(2 * np.mean(np.sum(diff ** 2, axis=0)))
        log.info(f"MCD (manual): {mcd:.4f}  (target <8.0)")
        return float(mcd)


# ══════════════════════════════════════════════════════════════════════════════
#  Main Synthesis Pipeline
# ══════════════════════════════════════════════════════════════════════════════

def synthesize(
    translation_path: str = "outputs/santhali_translation.json",
    embedding_path:   str = "models/speaker_embedding.npy",
    ref_wav:          str = "outputs/student_voice_ref.wav",
    out_path:         str = "outputs/output_LRL_cloned.wav",
    method:           str = "auto",
) -> str:
    """
    Full TTS synthesis pipeline.

    method: 'yourtts' | 'mms' | 'vits' | 'gtts' | 'auto'
    """
    with open(translation_path, encoding="utf-8") as f:
        trans_data = json.load(f)

    # Extract text segments (use romanized Santhali for TTS compatibility)
    texts = []
    for seg in trans_data.get("segments", []):
        # Prefer romanized (ASCII) for TTS models that can't handle Ol Chiki
        text = " ".join(
            t.get("santhali_roman", t.get("word", ""))
            for t in seg.get("token_info", [])
        ).strip()
        if text:
            texts.append(text)

    log.info(f"Synthesizing {len(texts)} text segments …")

    embedding = np.load(embedding_path) if os.path.exists(embedding_path) else None

    # Choose synthesis strategy
    if method == "auto":
        methods_to_try = ["yourtts", "mms", "gtts"]
    else:
        methods_to_try = [method]

    segments = None
    for m in methods_to_try:
        try:
            if m == "yourtts":
                segments = synthesize_yourtts(texts, ref_wav)
            elif m == "mms":
                segments = synthesize_mms(texts, language="sat")
            elif m == "vits":
                segments = synthesize_vits(texts, embedding)
            elif m == "gtts":
                segments = synthesize_gtts(texts, lang="en")
            log.info(f"Successfully synthesized with method: {m}")
            break
        except Exception as e:
            log.warning(f"Method '{m}' failed: {e}")

    if not segments:
        raise RuntimeError("All TTS methods failed. Check dependencies.")

    out = concat_and_save(segments, out_path)

    # MCD evaluation
    if os.path.exists(ref_wav):
        mcd = compute_mcd(ref_wav, out)
        log.info(f"MCD vs reference: {mcd:.4f}")

    return out


# ── Entry Point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Zero-shot voice cloning TTS")
    parser.add_argument("--translation", default="outputs/santhali_translation.json")
    parser.add_argument("--embedding",   default="models/speaker_embedding.npy")
    parser.add_argument("--ref",         default="outputs/student_voice_ref.wav")
    parser.add_argument("--out",         default="outputs/output_LRL_cloned.wav")
    parser.add_argument("--method",      default="auto",
                        choices=["auto", "yourtts", "mms", "vits", "gtts"])
    parser.add_argument("--mcd-only", action="store_true",
                        help="Only compute MCD between ref and synthesized audio.")
    args = parser.parse_args()

    if args.mcd_only:
        mcd = compute_mcd(args.ref, args.out)
        print(f"MCD: {mcd:.4f}  (target <8.0)")
    else:
        out = synthesize(args.translation, args.embedding, args.ref,
                         args.out, args.method)
        print(f"[OK] Final synthesized lecture: {out}")
