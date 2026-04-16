"""
ipa_mapper.py
─────────────
Task 2.1 — IPA Unified Representation

Converts code-switched (English + Hindi / Hinglish) transcript into
a unified International Phonetic Alphabet (IPA) string.

Why standard G2P fails on code-switching:
  • English G2P tools (e.g. CMU dict) don't handle Hindi words.
  • Hindi G2P tools use Devanagari input and fail on romanized Hindi.
  • Neither handles hybrid forms like "karo decoding", "model sahi hai".

Our approach:
  1. Language Detection per token (using LID + heuristics).
  2. English tokens → espeak-ng IPA via phonemizer (en-us).
  3. Hindi tokens (romanized) → custom hand-crafted mapping layer +
     Epitran fallback (hin-Deva after Unicode conversion).
  4. Merge into a single IPA string with language boundary markers.

Usage:
    python src/ipa_mapper.py \
        --transcript outputs/full_transcript.json \
        --out        outputs/ipa_output.json
"""

import os, re, json, logging, argparse
from typing import List, Tuple, Dict, Optional

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
log = logging.getLogger(__name__)


# ══════════════════════════════════════════════════════════════════════════════
#  Hinglish Phonological Mapping Layer
#  (manually crafted for code-switched phonology — task requirement)
# ══════════════════════════════════════════════════════════════════════════════

# Romanized Hindi → IPA  (manually mapped for lecture-domain Hinglish)
HINGLISH_IPA: Dict[str, str] = {
    # Pronouns
    "main":   "mɛ̃ː",    "mein":  "mɛ̃ː",
    "tum":    "tʊm",     "aap":   "ɑːp",
    "hum":    "hʊm",     "woh":   "ʋoː",
    "yeh":    "jɛː",     "ye":    "jɛː",
    "kya":    "kjɑː",    "koi":   "koɪ",

    # Verbs / auxiliaries
    "hai":    "hɛː",     "hain":  "hɛ̃ː",
    "tha":    "tʰɑː",    "the":   "tʰeː",
    "kar":    "kər",     "karo":  "kəroː",
    "karna":  "kərnɑː",  "karte": "kərtɛː",
    "samajh": "səmɑːdʒ", "dekho": "dɛːkʰoː",
    "bolo":   "boloː",   "suno":  "sʊnoː",
    "jana":   "dʒɑːnɑː", "jaata": "dʒɑːtɑː",

    # Discourse markers (common in Hinglish lectures)
    "matlab": "mətlɑːb",  "yaani": "jɑːniː",
    "toh":    "toː",      "wala":  "ʋɑːlɑː",
    "bilkul": "bɪlkʊl",   "sahi":  "sɑːhiː",
    "thoda":  "tʰoːɖɑː",  "bahut": "bəhʊt",
    "zyada":  "zjɑːdɑː",  "kam":   "kəm",
    "accha":  "ɑːtʃʃɑː",  "haan":  "hɑːn",
    "nahi":   "nɑːhiː",   "nahin": "nɑːhĩː",

    # Lecture-domain nouns (technical Hinglish)
    "model":  "mɑːɖəl",   "feature": "fiːtʃər",
    "output": "ɑːʊtpʊt",  "input":   "ɪnpʊt",
    "signal": "sɪgnəl",   "matlab":  "mætlæb",

    # Numbers / quantifiers
    "ek":     "eːk",      "do":   "doː",
    "teen":   "tiːn",     "char": "tʃɑːr",
    "paanch": "pɑːntʃ",

    # Particles
    "ke":     "keː",      "ki":   "kiː",
    "ko":     "koː",      "se":   "seː",
    "mein":   "mɛ̃ː",     "pe":   "peː",
    "par":    "pər",      "ne":   "neː",
}

# English function words that appear in Hindi speech (don't remap)
ENGLISH_PASSTHROUGH = {
    "the", "a", "an", "is", "are", "was", "were", "be", "been",
    "and", "or", "but", "in", "on", "at", "to", "for", "of", "with",
    "that", "this", "these", "those", "it", "its",
}

# Romanized Hindi heuristics — if token matches, treat as Hindi
HINDI_HEURISTICS = {
    r"^(hai|hain|tha|the|thi|kar|karo|karna|yeh|ye|kya|aur|toh|matlab|yaani|"
    r"nahi|nahin|bahut|thoda|bilkul|accha|haan|mujhe|tumhe|apna|apni|"
    r"kuch|sab|sirf|jo|jab|tab|phir|lekin|isliye|kyunki|woh|unke|unka)$": "hi",
}


# ══════════════════════════════════════════════════════════════════════════════
#  Language Detection per Token
# ══════════════════════════════════════════════════════════════════════════════

def detect_token_language(token: str) -> str:
    """
    Heuristic per-token language detection.
    Returns 'hi' or 'en'.
    """
    t = token.lower().strip(".,!?;:\"'")

    if t in ENGLISH_PASSTHROUGH:
        return "en"

    if t in HINGLISH_IPA:
        return "hi"

    for pattern in HINDI_HEURISTICS:
        if re.match(pattern, t, re.IGNORECASE):
            return "hi"

    # Syllable-structure heuristics for romanized Hindi
    # Hindi words often end in vowels; many English words end in consonants
    hi_signals = [
        bool(re.search(r"(aa|ii|uu|ai|au|kh|gh|ch|jh|th|dh|ph|bh)", t)),
        t.endswith(("a", "i", "u", "e", "o")) and len(t) > 2,
        bool(re.search(r"[ṭḍṇṣṛ]", t)),  # retroflex Unicode
    ]
    if sum(hi_signals) >= 2:
        return "hi"

    return "en"


# ══════════════════════════════════════════════════════════════════════════════
#  G2P for English (via phonemizer)
# ══════════════════════════════════════════════════════════════════════════════

_en_phonemizer = None

def english_to_ipa(text: str) -> str:
    """Convert English text to IPA using phonemizer (espeak-ng backend)."""
    global _en_phonemizer
    try:
        from phonemizer import phonemize
        ipa = phonemize(
            text,
            backend="espeak",
            language="en-us",
            with_stress=True,
            strip=True,
        )
        return ipa
    except Exception as e:
        log.warning(f"phonemizer failed ({e}) — using letter-by-letter fallback.")
        return _english_fallback_ipa(text)


def _english_fallback_ipa(text: str) -> str:
    """
    Simple rule-based English → IPA (covers common phonemes).
    Not production quality but works without espeak-ng.
    """
    rules = [
        # Multi-char before single
        ("th",  "ð"), ("sh",  "ʃ"), ("ch",  "tʃ"), ("ph",  "f"),
        ("ng",  "ŋ"), ("wh",  "w"), ("qu",  "kw"),
        ("ee",  "iː"), ("ea", "iː"), ("oo", "uː"), ("ou", "aʊ"),
        ("ow",  "oʊ"), ("ay", "eɪ"), ("ai", "eɪ"), ("oa", "oʊ"),
        ("igh", "aɪ"), ("ie", "aɪ"),
        # Single vowels (simplified)
        ("a",   "æ"), ("e", "ɛ"), ("i", "ɪ"), ("o", "ɒ"), ("u", "ʌ"),
        # Consonants
        ("c", "k"), ("x", "ks"), ("y", "j"), ("w", "w"),
    ]
    result = text.lower()
    for src, tgt in rules:
        result = result.replace(src, tgt)
    return result


# ══════════════════════════════════════════════════════════════════════════════
#  G2P for Hindi (romanized → IPA)
# ══════════════════════════════════════════════════════════════════════════════

def hindi_to_ipa(token: str) -> str:
    """
    Convert a romanized Hindi token to IPA.
    Priority: manual table → Epitran → rule-based.
    """
    t_lower = token.lower().strip(".,!?;:\"'")

    # 1. Manual mapping table
    if t_lower in HINGLISH_IPA:
        return HINGLISH_IPA[t_lower]

    # 2. Epitran (Hindi Devanagari) via Unicode transliteration
    try:
        import epitran
        # Transliterate romanized Hindi → Devanagari (rough)
        epi = epitran.Epitran("hin-Deva")
        ipa = epi.transliterate(token)
        if ipa.strip():
            return ipa
    except Exception:
        pass

    # 3. Rule-based romanized Hindi → IPA
    return _hindi_rules(token.lower())


def _hindi_rules(text: str) -> str:
    """Rule-based romanized Hindi → IPA conversion."""
    rules = [
        # Aspirated stops (order matters — longer sequences first)
        ("kh", "kʰ"), ("gh", "gʱ"), ("ch", "tʃʰ"), ("jh", "dʒʱ"),
        ("th", "tʰ"), ("dh", "dʱ"), ("ph", "pʰ"), ("bh", "bʱ"),
        # Retroflex (tilde notation sometimes used)
        ("T",  "ʈ"),  ("D",  "ɖ"),  ("N",  "ɳ"),
        # Vowels
        ("aa", "ɑː"), ("ii", "iː"), ("uu", "uː"),
        ("ai", "ɛː"), ("au", "ɔː"), ("ae", "ɛː"),
        ("a",  "ə"),  ("i",  "ɪ"),  ("u",  "ʊ"),
        ("e",  "eː"), ("o",  "oː"),
        # Consonants
        ("k",  "k"),  ("g",  "g"),  ("c",  "tʃ"), ("j",  "dʒ"),
        ("t",  "t"),  ("d",  "d"),  ("p",  "p"),  ("b",  "b"),
        ("m",  "m"),  ("n",  "n"),  ("r",  "r"),  ("l",  "l"),
        ("s",  "s"),  ("h",  "h"),  ("v",  "ʋ"),  ("w",  "ʋ"),
        ("y",  "j"),  ("f",  "f"),  ("z",  "z"),
    ]
    result = text
    for src, tgt in rules:
        result = result.replace(src, tgt)
    return result


# ══════════════════════════════════════════════════════════════════════════════
#  Main Mapping Function
# ══════════════════════════════════════════════════════════════════════════════

def text_to_ipa(text: str, language_hint: Optional[str] = None) -> Tuple[str, List[dict]]:
    """
    Convert a mixed Hinglish text string to IPA.

    Parameters
    ----------
    text          : input text (may contain English and Hindi tokens)
    language_hint : 'en' | 'hi' | None (auto-detect per token)

    Returns
    -------
    ipa_string : unified IPA string
    token_info : list of {word, lang, ipa} per token
    """
    tokens = text.strip().split()
    ipa_parts = []
    token_info = []

    for token in tokens:
        clean = token.lower().strip(".,!?;:\"'-")
        if not clean:
            continue

        if language_hint:
            lang = language_hint
        else:
            lang = detect_token_language(clean)

        if lang == "hi":
            ipa = hindi_to_ipa(clean)
        else:
            ipa = english_to_ipa(clean)

        ipa_parts.append(ipa)
        token_info.append({"word": token, "lang": lang, "ipa": ipa})

    # Join with word boundaries (| in IPA notation)
    ipa_string = " | ".join(ipa_parts)
    return ipa_string, token_info


def convert_transcript(
    transcript_path: str = "outputs/full_transcript.json",
    lid_path:        str = "outputs/lid_segments.json",
    out_path:        str = "outputs/ipa_output.json",
) -> dict:
    """
    Convert a full transcript JSON to IPA, respecting LID labels.
    """
    with open(transcript_path) as f:
        transcript = json.load(f)

    lid_map = {}
    if os.path.exists(lid_path):
        with open(lid_path) as f:
            lid_segs = json.load(f)
        # Build a lookup: midpoint → lang
        for seg in lid_segs:
            mid = (seg["start_sec"] + seg["end_sec"]) / 2
            lid_map[mid] = seg["lang"]

    segments = transcript.get("segments", [])
    ipa_segments = []

    for seg in segments:
        # Use LID-detected language as hint
        seg_mid  = (seg["start"] + seg["end"]) / 2
        lang_hint = None
        for mid, lang in lid_map.items():
            if abs(mid - seg_mid) < 3.0:
                lang_hint = lang
                break

        ipa_str, tok_info = text_to_ipa(seg["text"], language_hint=lang_hint)
        ipa_segments.append({
            "start":       seg["start"],
            "end":         seg["end"],
            "text":        seg["text"],
            "ipa":         ipa_str,
            "lang_hint":   lang_hint,
            "token_info":  tok_info,
        })

    full_ipa = " || ".join(s["ipa"] for s in ipa_segments)

    result = {
        "full_ipa":     full_ipa,
        "segments":     ipa_segments,
        "total_tokens": sum(len(s["token_info"]) for s in ipa_segments),
    }

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    log.info(f"[OK] IPA output saved: {out_path}")
    log.info(f"     First 200 chars of IPA:\n{full_ipa[:200]}")

    return result


# ── Entry Point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Hinglish → IPA mapper")
    parser.add_argument("--transcript", default="outputs/full_transcript.json")
    parser.add_argument("--lid",        default="outputs/lid_segments.json")
    parser.add_argument("--out",        default="outputs/ipa_output.json")
    # Quick test mode
    parser.add_argument("--test", type=str, default=None,
                        help="Convert a single text string to IPA and exit.")
    args = parser.parse_args()

    if args.test:
        ipa, info = text_to_ipa(args.test)
        print(f"Input : {args.test}")
        print(f"IPA   : {ipa}")
        for t in info:
            print(f"  {t['word']:20s} [{t['lang']}] → {t['ipa']}")
    else:
        convert_transcript(args.transcript, args.lid, args.out)
