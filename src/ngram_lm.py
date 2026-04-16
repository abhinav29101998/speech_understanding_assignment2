"""
ngram_lm.py
───────────
Task 1.2 (part A) — N-gram Language Model trained on the Speech Course Syllabus.

The N-gram LM scores technical term sequences so that the constrained decoder
in Whisper can upweight them.

Mathematical formulation
────────────────────────
For a sequence of tokens w₁, w₂, …, wₙ, the N-gram probability is:

    P(wₙ | w₁…wₙ₋₁) ≈ P(wₙ | wₙ₋(N-1)…wₙ₋₁)

With Kneser-Ney smoothing (Chen & Goodman, 1999):

    P_KN(wₙ | context) = max(C(context, wₙ) − d, 0) / C(context)
                        + λ(context) · P_KN(wₙ | shorter_context)

where d is the discount (≈ 0.75) and λ(context) is a normalisation weight.

Usage:
    python src/ngram_lm.py --corpus data/syllabus_corpus.txt --n 3
"""

import os, re, json, math, argparse, logging
from collections import defaultdict, Counter
from typing import List, Tuple, Dict, Optional
import numpy as np

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
log = logging.getLogger(__name__)

# Technical terms that MUST be prioritised during ASR decoding
TECHNICAL_TERMS = [
    # Core speech processing
    "stochastic", "cepstrum", "cepstral", "mel-frequency", "mfcc",
    "spectrogram", "fundamental frequency", "pitch", "formant",
    "fricative", "plosive", "phoneme", "phonetics", "allophone",
    "diphthong", "monophthong", "coarticulation", "prosody",
    # DSP
    "fourier", "discrete cosine transform", "dct", "windowing",
    "hamming", "hanning", "pre-emphasis", "liftering", "delta",
    "zero crossing rate", "energy", "autocorrelation",
    # Language models
    "trigram", "bigram", "unigram", "perplexity", "smoothing",
    "kneser-ney", "laplace", "backoff", "interpolation",
    "hidden markov model", "hmm", "viterbi", "forward-backward",
    "baum-welch", "emission probability", "transition probability",
    # Neural
    "neural network", "lstm", "gru", "attention", "transformer",
    "wav2vec", "hubert", "conformer", "connectionist temporal classification",
    "ctc", "sequence-to-sequence", "encoder decoder",
    # Evaluation
    "word error rate", "wer", "character error rate", "levenshtein",
    "edit distance", "alignment",
    # Code-switching
    "code switching", "hinglish", "language identification",
    "language boundary", "matrix language", "embedded language",
]


# ══════════════════════════════════════════════════════════════════════════════
#  Text Preprocessing
# ══════════════════════════════════════════════════════════════════════════════

def tokenize(text: str) -> List[str]:
    """Simple whitespace + punctuation tokenizer, lowercase."""
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s'-]", " ", text)
    tokens = text.split()
    return ["<s>"] + tokens + ["</s>"]


def load_corpus(path: str) -> List[List[str]]:
    """Load corpus file, one sentence per line. Return list of token lists."""
    sentences = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                sentences.append(tokenize(line))
    log.info(f"Loaded {len(sentences)} sentences from {path}")
    return sentences


# ══════════════════════════════════════════════════════════════════════════════
#  N-gram Model with Kneser-Ney Smoothing
# ══════════════════════════════════════════════════════════════════════════════

class NGramLM:
    """
    N-gram language model with Kneser-Ney smoothing.

    Attributes
    ----------
    n       : order (e.g. 3 for trigram)
    d       : Kneser-Ney discount (default 0.75)
    vocab   : set of known tokens
    counts  : dict mapping n-gram tuple → count
    context_counts : dict mapping (n-1)-gram context → count
    """

    def __init__(self, n: int = 3, discount: float = 0.75):
        self.n        = n
        self.d        = discount
        self.vocab: set = set()
        self.counts:   Dict[Tuple, int] = defaultdict(int)
        self.ctx_cts:  Dict[Tuple, int] = defaultdict(int)
        self.uni_cts:  Counter = Counter()
        # Continuation counts for KN (unigram continuation probability)
        self.cont_cts: Counter = Counter()   # how many contexts word w completes

    def _ngrams(self, tokens: List[str], k: int):
        for i in range(len(tokens) - k + 1):
            yield tuple(tokens[i: i + k])

    def train(self, sentences: List[List[str]]):
        log.info(f"Training {self.n}-gram LM …")
        for sent in sentences:
            for tok in sent:
                self.vocab.add(tok)
                self.uni_cts[tok] += 1
            for k in range(2, self.n + 1):
                for gram in self._ngrams(sent, k):
                    self.counts[gram]      += 1
                    self.ctx_cts[gram[:-1]] += 1
            # Kneser-Ney continuation counts (bigram level)
            for gram in self._ngrams(sent, 2):
                self.cont_cts[gram[-1]] += 1  # how many unique w_{i-1} precede w_i

        self.total_tokens = sum(self.uni_cts.values())
        log.info(f"Vocab size: {len(self.vocab)}  |  Total tokens: {self.total_tokens}")

    def _kn_prob(self, word: str, context: Tuple[str, ...]) -> float:
        """
        Recursively compute Kneser-Ney probability.
        P_KN(w | context) — base case is unigram continuation probability.
        """
        if len(context) == 0:
            # Unigram continuation probability
            total_cont = sum(self.cont_cts.values()) or 1
            return (self.cont_cts.get(word, 0) + 1e-9) / total_cont

        gram      = context + (word,)
        gram_cnt  = self.counts.get(gram, 0)
        ctx_cnt   = self.ctx_cts.get(context, 0)

        if ctx_cnt == 0:
            return self._kn_prob(word, context[1:])

        # Number of unique words that follow context (for λ)
        n_unique_follow = sum(
            1 for g, c in self.counts.items()
            if len(g) == len(context) + 1 and g[:-1] == context and c > 0
        )

        prob_first   = max(gram_cnt - self.d, 0.0) / ctx_cnt
        lambda_ctx   = (self.d * n_unique_follow) / ctx_cnt
        prob_lower   = self._kn_prob(word, context[1:])

        return prob_first + lambda_ctx * prob_lower

    def log_prob(self, word: str, context: Tuple[str, ...]) -> float:
        """log P(word | context), using (n-1)-gram context."""
        ctx = context[-(self.n - 1):]
        p   = self._kn_prob(word, ctx)
        return math.log(p + 1e-300)

    def score_sequence(self, tokens: List[str]) -> float:
        """Return sum of log-probs for a token sequence (log-likelihood)."""
        total = 0.0
        for i in range(1, len(tokens)):
            ctx   = tuple(tokens[max(0, i - self.n + 1): i])
            total += self.log_prob(tokens[i], ctx)
        return total

    def perplexity(self, sentences: List[List[str]]) -> float:
        total_log = 0.0
        total_tok  = 0
        for sent in sentences:
            for i in range(1, len(sent)):
                ctx = tuple(sent[max(0, i - self.n + 1): i])
                total_log += self.log_prob(sent[i], ctx)
                total_tok  += 1
        ppl = math.exp(-total_log / total_tok)
        return ppl

    def save(self, path: str):
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        data = {
            "n":       self.n,
            "d":       self.d,
            "vocab":   list(self.vocab),
            "counts":  {str(k): v for k, v in self.counts.items()},
            "ctx_cts": {str(k): v for k, v in self.ctx_cts.items()},
            "uni_cts": dict(self.uni_cts),
            "cont_cts": dict(self.cont_cts),
            "total_tokens": self.total_tokens,
        }
        with open(path, "w") as f:
            json.dump(data, f)
        log.info(f"N-gram LM saved: {path}")

    @classmethod
    def load(cls, path: str) -> "NGramLM":
        with open(path) as f:
            data = json.load(f)
        lm = cls(n=data["n"], discount=data["d"])
        lm.vocab  = set(data["vocab"])
        lm.counts = {eval(k): v for k, v in data["counts"].items()}
        lm.ctx_cts = {eval(k): v for k, v in data["ctx_cts"].items()}
        lm.uni_cts = Counter(data["uni_cts"])
        lm.cont_cts = Counter(data["cont_cts"])
        lm.total_tokens = data["total_tokens"]
        log.info(f"N-gram LM loaded: {path}  (n={lm.n})")
        return lm


# ══════════════════════════════════════════════════════════════════════════════
#  Logit-Bias Table
# ══════════════════════════════════════════════════════════════════════════════

def build_logit_bias_table(
    lm: NGramLM,
    tokenizer,               # Whisper tokenizer (tiktoken)
    boost: float = 3.0,      # additive logit boost for technical terms
) -> Dict[int, float]:
    """
    Build a {token_id: logit_bias} dictionary for Whisper's decoder.

    Mathematical formulation:
        logit_bias[t] = boost × I[token t ∈ technical_vocab]
                       + α × log P_KN(t | lm)

    where α scales the LM contribution relative to Whisper's own LM head.
    """
    bias_table: Dict[int, float] = {}
    alpha = 1.0

    for term in TECHNICAL_TERMS:
        words = term.lower().split()
        for word in words:
            # Get all Whisper token IDs that could map to this word
            try:
                ids = tokenizer.encode(" " + word)   # leading space for BPE
                for tid in ids:
                    ctx = tuple(tokenizer.decode([tid]).strip().split())
                    lm_score = lm.log_prob(word, ctx) if len(lm.vocab) > 0 else 0.0
                    bias = boost + alpha * max(lm_score, -10.0)
                    bias_table[tid] = max(bias_table.get(tid, 0.0), bias)
            except Exception:
                pass

    log.info(f"Logit-bias table: {len(bias_table)} tokens boosted.")
    return bias_table


# ══════════════════════════════════════════════════════════════════════════════
#  Syllabus Corpus (embedded — used when the file is missing)
# ══════════════════════════════════════════════════════════════════════════════

SYLLABUS_TEXT = """
Introduction to speech processing and speech understanding systems.
The acoustic model maps features to phonemes using hidden markov models.
Mel frequency cepstral coefficients are computed from the power spectrum.
The cepstrum is the inverse fourier transform of the log spectrum.
Stochastic models include gaussian mixture models and hmm.
The viterbi algorithm finds the most likely state sequence.
Constrained beam search uses a language model to restrict hypotheses.
Pre-emphasis filter boosts high frequency components before windowing.
Hamming window reduces spectral leakage in short time fourier transform.
Fundamental frequency or pitch is extracted using autocorrelation.
Formant frequencies define the vocal tract resonances.
Coarticulation refers to the influence of adjacent phonemes.
Prosody includes pitch duration and loudness variations.
Code switching occurs when speakers alternate between two languages.
Language identification classifies speech segments by language.
Kneser-ney smoothing addresses the data sparsity problem in n-gram models.
Word error rate measures the edit distance between hypothesis and reference.
The connectionist temporal classification loss handles sequence alignment.
Wav2vec learns speech representations through self-supervised learning.
The conformer architecture combines convolution and attention mechanisms.
Speaker verification uses d-vector or x-vector embeddings.
Dynamic time warping aligns sequences with different temporal lengths.
Zero-shot voice cloning synthesises speech in a new voice without retraining.
The mel-cepstral distortion measures synthesized audio quality.
Adversarial examples are imperceptible perturbations that fool classifiers.
Fast gradient sign method computes perturbations from the gradient sign.
Anti-spoofing systems detect synthetic or replayed speech.
Linear frequency cepstral coefficients differ from mfcc in filter spacing.
Equal error rate balances false acceptance and false rejection rates.
Spectral subtraction removes stationary noise from speech signals.
"""


def ensure_corpus(path: str = "data/syllabus_corpus.txt"):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    if not os.path.exists(path):
        with open(path, "w") as f:
            f.write(SYLLABUS_TEXT.strip())
        log.info(f"Syllabus corpus written: {path}")


# ── Entry Point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train N-gram LM on syllabus")
    parser.add_argument("--corpus", default="data/syllabus_corpus.txt")
    parser.add_argument("--save",   default="models/ngram_lm.json")
    parser.add_argument("--n",      type=int, default=3)
    args = parser.parse_args()

    ensure_corpus(args.corpus)
    sents = load_corpus(args.corpus)

    lm = NGramLM(n=args.n)
    lm.train(sents)

    # Split for perplexity
    split = max(1, int(0.9 * len(sents)))
    ppl = lm.perplexity(sents[split:])
    log.info(f"Perplexity on held-out: {ppl:.2f}")

    lm.save(args.save)
    log.info(f"[OK] N-gram LM ({args.n}-gram) saved to {args.save}")
