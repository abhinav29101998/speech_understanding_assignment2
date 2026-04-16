"""
constrained_decoding.py
───────────────────────
Task 1.2 — Constrained Decoding with N-gram Logit Bias

Implements two mechanisms on top of Whisper:
  1. Logit Bias        : add a constant bias to token logits for technical terms
  2. N-gram LM Fusion  : shallow fusion of Whisper's LM with the syllabusN-gram LM

Mathematical formulation
────────────────────────
At each decoding step t, Whisper produces logits l_i for each vocabulary token i.
We modify the logits before sampling/beam search:

    l̃_i = l_i + β_i + α · log P_KN(w_i | context)

where:
  β_i   = logit bias for token i (from the bias table, 0 if not a technical term)
  α     = interpolation weight for the N-gram LM (shallow fusion)
  P_KN  = Kneser-Ney N-gram probability

Beam search keeps the top-K hypotheses at each step.

Usage:
    python src/constrained_decoding.py \
        --audio outputs/denoised_segment.wav \
        --lm    models/ngram_lm.json \
        --out   outputs/transcript.json
"""

import os, json, logging
import numpy as np
import torch
import torch.nn.functional as F
from typing import Dict, List, Optional

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
log = logging.getLogger(__name__)

# Shallow-fusion weight for N-gram LM
ALPHA_FUSION = 0.3
LOGIT_BOOST  = 4.0   # additive boost for technical term tokens


# ══════════════════════════════════════════════════════════════════════════════
#  Logit Processor (applied at every decoding step)
# ══════════════════════════════════════════════════════════════════════════════

class NGramLogitProcessor:
    """
    Applies N-gram logit bias to Whisper token logits at each decoding step.

    This is injected into Whisper's beam search via a hook on the decoder.
    """

    def __init__(
        self,
        bias_table: Dict[int, float],
        lm=None,
        alpha: float = ALPHA_FUSION,
        tokenizer=None,
    ):
        self.bias_table = bias_table   # {token_id: bias_value}
        self.lm         = lm
        self.alpha      = alpha
        self.tokenizer  = tokenizer
        self.context: List[str] = []

    def __call__(self, token_ids: List[int], logits: torch.Tensor) -> torch.Tensor:
        """
        Modify logits in-place.

        Parameters
        ----------
        token_ids : list of already-decoded token IDs (for N-gram context)
        logits    : (vocab_size,) tensor

        Returns
        -------
        Modified logits tensor.
        """
        # ── Step 1: Logit bias for technical terms ────────────────────────────
        for tid, bias in self.bias_table.items():
            if tid < logits.shape[-1]:
                logits[..., tid] += bias

        # ── Step 2: Shallow N-gram LM fusion ─────────────────────────────────
        if self.lm is not None and self.tokenizer is not None and len(token_ids) > 0:
            # Build context from last (n-1) tokens
            ctx_words = []
            for tid in token_ids[-(self.lm.n - 1):]:
                try:
                    word = self.tokenizer.decode([tid]).strip().lower()
                    if word:
                        ctx_words.append(word)
                except Exception:
                    pass
            ctx_tuple = tuple(ctx_words)

            # Only apply for a subset of vocabulary to keep it efficient
            # (applying to all 50k tokens would be too slow)
            top_k_ids = logits.topk(200).indices.tolist()
            for tid in top_k_ids:
                try:
                    word = self.tokenizer.decode([tid]).strip().lower()
                    if word and word in self.lm.vocab:
                        lm_lp = self.lm.log_prob(word, ctx_tuple)
                        logits[tid] += self.alpha * lm_lp
                except Exception:
                    pass

        return logits


# ══════════════════════════════════════════════════════════════════════════════
#  Constrained Beam Search Hook
# ══════════════════════════════════════════════════════════════════════════════

class ConstrainedWhisperDecoder:
    """
    Wraps Whisper's generation with custom logit processing.

    Usage:
        decoder = ConstrainedWhisperDecoder(model, tokenizer, bias_table, lm)
        result  = decoder.transcribe(audio_array)
    """

    def __init__(self, model, tokenizer, bias_table, lm=None,
                 beam_size: int = 5, alpha: float = ALPHA_FUSION):
        self.model      = model
        self.tokenizer  = tokenizer
        self.processor  = NGramLogitProcessor(bias_table, lm, alpha, tokenizer)
        self.beam_size  = beam_size

    def transcribe(
        self,
        audio: np.ndarray,
        language: str = "en",
        task: str = "transcribe",
    ) -> dict:
        """
        Transcribe audio using Whisper with constrained logit bias.

        Returns dict with keys: text, segments, language
        """
        import whisper

        log.info(f"Running constrained decoding (beam={self.beam_size}, α={self.processor.alpha}) …")

        # Compute mel spectrogram
        mel = whisper.log_mel_spectrogram(whisper.pad_or_trim(audio))
        mel = mel.to(self.model.device)

        # Decode options — pass our logit_filters list
        options = whisper.DecodingOptions(
            language=language,
            task=task,
            beam_size=self.beam_size,
            # logit_filters is not a public Whisper API param, so we monkey-patch
        )

        # ── Monkey-patch: inject our processor into Whisper's decode loop ──
        original_inference = self.model.decoder.forward

        def patched_forward(x, xa, mask=None, kv_cache=None):
            logits = original_inference(x, xa, mask=mask, kv_cache=kv_cache)
            # logits shape: (batch, seq, vocab)
            # Apply processor to last position
            token_ids = x[0].tolist() if x.dim() == 2 else []
            logits[:, -1, :] = self.processor(token_ids, logits[:, -1, :].clone())
            return logits

        self.model.decoder.forward = patched_forward

        try:
            result = whisper.decode(self.model, mel, options)
            text   = result.text
            # Restore
            self.model.decoder.forward = original_inference
        except Exception as e:
            self.model.decoder.forward = original_inference
            log.error(f"Constrained decode failed: {e}. Using standard decode.")
            result = whisper.decode(self.model, mel, options)
            text   = result.text

        log.info(f"Transcript (first 200 chars): {text[:200]}")
        return {"text": text, "language": result.language}


# ══════════════════════════════════════════════════════════════════════════════
#  Full-Audio Transcription with Sliding Window
# ══════════════════════════════════════════════════════════════════════════════

def transcribe_full_audio(
    audio: np.ndarray,
    sr: int,
    model,
    tokenizer,
    bias_table: Dict[int, float],
    lm=None,
    window_sec: float = 30.0,
    stride_sec: float = 25.0,
) -> List[dict]:
    """
    Transcribe a long audio file in overlapping 30-second windows
    (Whisper's native chunk size) with constrained decoding.

    Returns list of {start, end, text, lang} segments.
    """
    decoder   = ConstrainedWhisperDecoder(model, tokenizer, bias_table, lm)
    hop       = int(stride_sec * sr)
    win       = int(window_sec * sr)
    segments  = []
    offset    = 0

    while offset < len(audio):
        chunk     = audio[offset: offset + win]
        start_sec = offset / sr
        end_sec   = (offset + len(chunk)) / sr

        if len(chunk) < sr:  # skip sub-second chunks
            break

        # Pad to 30s if needed
        chunk_padded = np.pad(chunk, (0, max(0, win - len(chunk))))

        result = decoder.transcribe(chunk_padded)
        segments.append({
            "start": round(start_sec, 3),
            "end":   round(end_sec,   3),
            "text":  result["text"].strip(),
            "lang":  result["language"],
        })
        offset += hop

    log.info(f"Transcription complete: {len(segments)} windows.")
    return segments


# ── Entry Point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import argparse, soundfile as sf
    from ngram_lm import NGramLM, ensure_corpus, load_corpus, build_logit_bias_table

    parser = argparse.ArgumentParser(description="Constrained Whisper decoding")
    parser.add_argument("--audio", default="outputs/denoised_segment.wav")
    parser.add_argument("--lm",    default="models/ngram_lm.json")
    parser.add_argument("--out",   default="outputs/transcript_constrained.json")
    parser.add_argument("--model", default="large-v3")
    args = parser.parse_args()

    import whisper
    log.info(f"Loading Whisper {args.model} …")
    wmodel = whisper.load_model(args.model)
    wtoken = wmodel.tokenizer if hasattr(wmodel, "tokenizer") else \
             whisper.tokenizer.get_tokenizer(wmodel.is_multilingual)

    # Load or train N-gram LM
    if os.path.exists(args.lm):
        lm = NGramLM.load(args.lm)
    else:
        ensure_corpus()
        sents = load_corpus("data/syllabus_corpus.txt")
        lm = NGramLM(n=3)
        lm.train(sents)
        lm.save(args.lm)

    bias_table = build_logit_bias_table(lm, wtoken, boost=LOGIT_BOOST)

    audio, sr = sf.read(args.audio, dtype="float32")
    if audio.ndim == 2:
        audio = audio.mean(axis=1)

    segments = transcribe_full_audio(audio, sr, wmodel, wtoken, bias_table, lm)

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(segments, f, indent=2, ensure_ascii=False)
    log.info(f"[OK] Transcript saved: {args.out}")

    # Print full text
    full_text = " ".join(s["text"] for s in segments)
    print(f"\nFull transcript:\n{full_text}\n")
