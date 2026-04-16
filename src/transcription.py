"""
transcription.py
────────────────
Task 1.2 — Main transcription module.

Orchestrates:
  1. Load denoised audio
  2. Run Whisper-large-v3 with constrained beam search + N-gram logit bias
  3. Compute WER for English and Hindi segments (using LID labels)
  4. Save full transcript JSON

Usage:
    python src/transcription.py \
        --audio   outputs/denoised_segment.wav \
        --lid_out outputs/lid_segments.json \
        --out     outputs/full_transcript.json
"""

import os, json, logging, argparse
import numpy as np

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
log = logging.getLogger(__name__)

WHISPER_MODEL = "large-v3"


def compute_wer(hypothesis: str, reference: str) -> float:
    """
    Word Error Rate = (S + D + I) / N
    using jiwer library (falls back to manual implementation).
    """
    try:
        from jiwer import wer
        return wer(reference, hypothesis)
    except ImportError:
        # Manual Levenshtein WER
        hyp_words = hypothesis.lower().split()
        ref_words = reference.lower().split()
        d = _levenshtein(ref_words, hyp_words)
        return d / max(len(ref_words), 1)


def _levenshtein(ref: list, hyp: list) -> int:
    n, m = len(ref), len(hyp)
    dp = list(range(m + 1))
    for i in range(1, n + 1):
        new_dp = [i] + [0] * m
        for j in range(1, m + 1):
            if ref[i - 1] == hyp[j - 1]:
                new_dp[j] = dp[j - 1]
            else:
                new_dp[j] = 1 + min(dp[j], new_dp[j - 1], dp[j - 1])
        dp = new_dp
    return dp[m]


def split_by_language(
    transcript_segments: list,
    lid_segments: list,
) -> dict:
    """
    Split transcribed text into English and Hindi portions
    based on LID timestamps.

    Returns {"en": [text, …], "hi": [text, …]}
    """
    lang_texts = {"en": [], "hi": []}

    for seg in transcript_segments:
        seg_mid = (seg["start"] + seg["end"]) / 2
        # Find the LID label for the midpoint of this segment
        lang = "en"  # default
        for lid in lid_segments:
            if lid["start_sec"] <= seg_mid <= lid["end_sec"]:
                lang = lid["lang"]
                break

        if lang in lang_texts:
            lang_texts[lang].append(seg["text"])

    return lang_texts


def transcribe(
    audio_path: str = "outputs/denoised_segment.wav",
    lid_path:   str = "outputs/lid_segments.json",
    lm_path:    str = "models/ngram_lm.json",
    out_path:   str = "outputs/full_transcript.json",
    model_name: str = WHISPER_MODEL,
) -> dict:

    import whisper, soundfile as sf
    from ngram_lm import NGramLM, ensure_corpus, load_corpus, build_logit_bias_table
    from constrained_decoding import transcribe_full_audio

    # ── Load audio ────────────────────────────────────────────────────────────
    log.info(f"Loading audio: {audio_path}")
    audio, sr = sf.read(audio_path, dtype="float32")
    if audio.ndim == 2:
        audio = audio.mean(axis=1)
    log.info(f"  Duration: {len(audio)/sr:.1f}s  | SR: {sr} Hz")

    # ── Load Whisper ──────────────────────────────────────────────────────────
    log.info(f"Loading Whisper {model_name} …")
    wmodel = whisper.load_model(model_name)
    wtoken = whisper.tokenizer.get_tokenizer(
        wmodel.is_multilingual, language="en", task="transcribe"
    )

    # ── Load / train N-gram LM ────────────────────────────────────────────────
    if os.path.exists(lm_path):
        lm = NGramLM.load(lm_path)
    else:
        log.info("Training N-gram LM from syllabus …")
        ensure_corpus()
        sents = load_corpus("data/syllabus_corpus.txt")
        lm = NGramLM(n=3)
        lm.train(sents)
        lm.save(lm_path)

    bias_table = build_logit_bias_table(lm, wtoken, boost=4.0)

    # ── Constrained decoding ──────────────────────────────────────────────────
    segments = transcribe_full_audio(
        audio, sr, wmodel, wtoken, bias_table, lm,
        window_sec=30.0, stride_sec=25.0
    )

    # ── Language split ────────────────────────────────────────────────────────
    lid_segments = []
    if os.path.exists(lid_path):
        with open(lid_path) as f:
            lid_segments = json.load(f)

    lang_texts = split_by_language(segments, lid_segments)
    full_text  = " ".join(s["text"] for s in segments)

    result = {
        "full_text":      full_text,
        "segments":       segments,
        "en_text":        " ".join(lang_texts["en"]),
        "hi_text":        " ".join(lang_texts["hi"]),
        "num_segments":   len(segments),
        "duration_sec":   len(audio) / sr,
    }

    # ── WER (if reference transcripts exist) ──────────────────────────────────
    en_ref_path = "data/ref_en.txt"
    hi_ref_path = "data/ref_hi.txt"

    if os.path.exists(en_ref_path):
        with open(en_ref_path) as f:
            ref_en = f.read()
        wer_en = compute_wer(result["en_text"], ref_en)
        result["wer_en"] = round(wer_en, 4)
        log.info(f"WER (English): {wer_en:.2%}  (target <15%)")

    if os.path.exists(hi_ref_path):
        with open(hi_ref_path) as f:
            ref_hi = f.read()
        wer_hi = compute_wer(result["hi_text"], ref_hi)
        result["wer_hi"] = round(wer_hi, 4)
        log.info(f"WER (Hindi): {wer_hi:.2%}  (target <25%)")

    # ── Save ──────────────────────────────────────────────────────────────────
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    log.info(f"[OK] Transcript saved: {out_path}")
    log.info(f"     Full text ({len(full_text.split())} words):\n{full_text[:300]}…")

    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--audio",   default="outputs/denoised_segment.wav")
    parser.add_argument("--lid_out", default="outputs/lid_segments.json")
    parser.add_argument("--out",     default="outputs/full_transcript.json")
    parser.add_argument("--model",   default=WHISPER_MODEL)
    args = parser.parse_args()
    transcribe(args.audio, args.lid_out, out_path=args.out, model_name=args.model)
