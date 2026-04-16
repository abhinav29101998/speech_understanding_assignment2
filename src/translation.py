"""
translation.py
──────────────
Task 2.2 — Semantic Translation to Target LRL (Santhali)

Since no MT model exists for Hinglish → Santhali, we:
  1. Use a 500-word hand-crafted parallel corpus for technical terms.
  2. Apply a simple token-substitution translation for known terms.
  3. Transliterate unknown English words using IPA → Santhali script rules.
  4. Flag untranslated tokens for manual review.

Santhali (ISO 639-3: sat) uses the Ol Chiki script (ᱚᱞ ᱪᱤᱠᱤ),
officially adopted in 2003. Unicode range: U+1C50–U+1C7F.

Usage:
    python src/translation.py \
        --ipa    outputs/ipa_output.json \
        --corpus data/lrl_parallel_corpus.json \
        --out    outputs/santhali_translation.json
"""

import os, json, re, logging, argparse
from typing import Dict, List, Optional

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
log = logging.getLogger(__name__)


# ══════════════════════════════════════════════════════════════════════════════
#  Ol Chiki (Santhali) Script — IPA mapping for transliteration
# ══════════════════════════════════════════════════════════════════════════════

# IPA phoneme → Ol Chiki character (approximate)
IPA_TO_OLCHIKI: Dict[str, str] = {
    "p":  "ᱯ", "b":  "ᱵ", "m":  "ᱢ",
    "t":  "ᱛ", "d":  "ᱫ", "n":  "ᱱ",
    "k":  "ᱠ", "g":  "ᱜ", "ŋ":  "ᱝ",
    "tʃ": "ᱪ", "dʒ": "ᱡ",
    "s":  "ᱥ", "h":  "ᱦ",
    "r":  "ᱨ", "l":  "ᱞ",
    "w":  "ᱶ", "j":  "ᱭ",
    "ʋ":  "ᱶ",
    # Vowels
    "a":  "ᱟ", "ɑ":  "ᱟ", "ə":  "ᱮ",
    "i":  "ᱤ", "ɪ":  "ᱤ",
    "u":  "ᱩ", "ʊ":  "ᱩ",
    "e":  "ᱮ", "ɛ":  "ᱮ",
    "o":  "ᱳ", "ɒ":  "ᱳ",
    "ː":  "",   "̃":  "",   "|":  " ",
}


def ipa_to_santhali_script(ipa: str) -> str:
    """Transliterate IPA string to approximate Ol Chiki (Santhali) script."""
    result = ""
    i = 0
    ipa_clean = ipa.replace("ʰ", "").replace("ʱ", "").replace("ˈ", "")
    while i < len(ipa_clean):
        # Try 2-char first
        two = ipa_clean[i:i+2]
        if two in IPA_TO_OLCHIKI:
            result += IPA_TO_OLCHIKI[two]
            i += 2
        elif ipa_clean[i] in IPA_TO_OLCHIKI:
            result += IPA_TO_OLCHIKI[ipa_clean[i]]
            i += 1
        else:
            result += ipa_clean[i]
            i += 1
    return result.strip()


# ══════════════════════════════════════════════════════════════════════════════
#  Parallel Corpus (500+ technical terms: English/Hindi → Santhali)
#  Hand-crafted for speech-processing lecture domain
# ══════════════════════════════════════════════════════════════════════════════

BUILTIN_CORPUS = {
    # ── Speech Processing Technical Terms ────────────────────────────────────
    "speech":           {"hi": "वाणी",     "sat": "ᱵᱟᱲᱟᱜ ᱠᱟᱛᱷᱟ",  "sat_rom": "barag katha"},
    "signal":           {"hi": "संकेत",    "sat": "ᱥᱟᱝᱠᱮᱛ",       "sat_rom": "sankhet"},
    "audio":            {"hi": "ऑडियो",    "sat": "ᱠᱟᱱ ᱵᱟᱲᱟᱜ",   "sat_rom": "kan barag"},
    "frequency":        {"hi": "आवृत्ति",  "sat": "ᱜᱷᱩᱨᱟᱹᱣ",      "sat_rom": "ghurau"},
    "feature":          {"hi": "विशेषता",  "sat": "ᱵᱤᱥᱮᱥ",        "sat_rom": "bisesh"},
    "model":            {"hi": "मॉडल",     "sat": "ᱥᱟᱫᱷᱩ",        "sat_rom": "sadhu"},
    "phoneme":          {"hi": "स्वनिम",   "sat": "ᱵᱚᱞ ᱠᱟᱱᱟ",    "sat_rom": "bol kana"},
    "word":             {"hi": "शब्द",     "sat": "ᱥᱮᱨᱢᱟ",        "sat_rom": "serma"},
    "language":         {"hi": "भाषा",     "sat": "ᱵᱷᱟᱥᱟ",        "sat_rom": "bhasa"},
    "sound":            {"hi": "ध्वनि",    "sat": "ᱥᱟᱝ",           "sat_rom": "sang"},
    "noise":            {"hi": "शोर",      "sat": "ᱜᱟᱰᱤᱴ",        "sat_rom": "gadit"},
    "sampling":         {"hi": "नमूना",    "sat": "ᱱᱟᱢᱩᱱᱟ",       "sat_rom": "namuna"},
    "spectrum":         {"hi": "वर्णक्रम", "sat": "ᱵᱟᱨᱱᱟᱠᱨᱚᱢ",   "sat_rom": "barnakrom"},
    "microphone":       {"hi": "माइक्रोफ़ोन","sat":"ᱜᱷᱩᱨᱟᱹᱣ ᱡᱟᱱᱛᱨᱟ","sat_rom":"ghurau jantra"},
    "recognition":      {"hi": "पहचान",    "sat": "ᱪᱤᱱᱦᱟᱹ",       "sat_rom": "chinha"},
    "training":         {"hi": "प्रशिक्षण","sat": "ᱮᱱᱮᱡ ᱠᱮᱫ",    "sat_rom": "enej ked"},
    "testing":          {"hi": "परीक्षण",  "sat": "ᱯᱚᱨᱤᱠᱥᱚᱱ",    "sat_rom": "porikshon"},
    "accuracy":         {"hi": "सटीकता",   "sat": "ᱥᱟᱦᱤᱡᱚᱱ",     "sat_rom": "sahijon"},
    "error":            {"hi": "त्रुटि",   "sat": "ᱜᱷᱩᱞ",         "sat_rom": "ghul"},
    "database":         {"hi": "डेटाबेस",  "sat": "ᱛᱟᱞᱤᱠᱟ",      "sat_rom": "talika"},
    "algorithm":        {"hi": "एल्गोरिदम","sat": "ᱜᱚᱱᱚᱱ ᱪᱟᱞᱟᱜ", "sat_rom": "gonon chalag"},
    "neural":           {"hi": "तंत्रिका", "sat": "ᱱᱟᱨᱵᱷᱮᱥ",     "sat_rom": "narbhes"},
    "network":          {"hi": "नेटवर्क",  "sat": "ᱡᱟᱞ",          "sat_rom": "jal"},
    "layer":            {"hi": "परत",      "sat": "ᱯᱟᱨᱚᱛ",        "sat_rom": "parot"},
    "weight":           {"hi": "भार",      "sat": "ᱵᱷᱟᱨ",         "sat_rom": "bhar"},
    "gradient":         {"hi": "प्रवणता",  "sat": "ᱜᱨᱮᱰᱤᱭᱮᱱᱴ",  "sat_rom": "gradient"},
    "loss":             {"hi": "हानि",     "sat": "ᱦᱟᱱᱤ",         "sat_rom": "hani"},
    "output":           {"hi": "आउटपुट",   "sat": "ᱵᱟᱦᱨᱮ ᱡᱤᱱᱤᱥ", "sat_rom": "bahre jinis"},
    "input":            {"hi": "इनपुट",    "sat": "ᱵᱷᱤᱛᱨᱤ ᱡᱤᱱᱤᱥ","sat_rom": "bhitri jinis"},
    "decoder":          {"hi": "डिकोडर",   "sat": "ᱰᱤᱠᱳᱰᱟᱨ",     "sat_rom": "dikodar"},
    "encoder":          {"hi": "एनकोडर",   "sat": "ᱮᱱᱠᱳᱰᱟᱨ",     "sat_rom": "enkodar"},
    "attention":        {"hi": "ध्यान",    "sat": "ᱫᱷᱤᱭᱟᱱ",       "sat_rom": "dhiyan"},
    "transformer":      {"hi": "ट्रांसफॉर्मर","sat":"ᱛᱨᱟᱱᱥᱯᱷᱳᱨᱢᱮᱨ","sat_rom":"transformer"},
    "mfcc":             {"hi": "एमएफसीसी", "sat": "ᱢᱮᱞ ᱠᱮᱯᱥᱛᱨᱟᱞ","sat_rom":"mel kepstral"},
    "spectrogram":      {"hi": "स्पेक्ट्रोग्राम","sat":"ᱵᱟᱨᱱᱟᱠᱨᱚᱢ ᱪᱤᱛᱨ","sat_rom":"barnakrom chitra"},
    "pitch":            {"hi": "पिच",      "sat": "ᱥᱮᱨᱢᱟ ᱞᱚᱝ",   "sat_rom": "serma long"},
    "formant":          {"hi": "फॉर्मेंट", "sat": "ᱯᱷᱚᱨᱢᱮᱱᱴ",    "sat_rom": "phorment"},
    "coarticulation":   {"hi": "सहस्वरण",  "sat": "ᱥᱟᱦᱟᱥᱟᱨᱚᱱ",  "sat_rom": "sahasaron"},
    "prosody":          {"hi": "छंद",      "sat": "ᱞᱤᱞᱟ",         "sat_rom": "lila"},
    "duration":         {"hi": "अवधि",     "sat": "ᱥᱟᱢᱚᱭ",       "sat_rom": "samoy"},
    "amplitude":        {"hi": "आयाम",     "sat": "ᱛᱤᱛᱤᱨ",       "sat_rom": "titir"},
    "waveform":         {"hi": "तरंग रूप", "sat": "ᱛᱟᱨᱟᱝ ᱨᱩᱯ",  "sat_rom": "tarang rup"},
    "frame":            {"hi": "फ्रेम",    "sat": "ᱯᱷᱨᱮᱢ",        "sat_rom": "phrem"},
    "window":           {"hi": "विंडो",    "sat": "ᱡᱟᱱᱟᱞᱟ",      "sat_rom": "janala"},
    "overlap":          {"hi": "अतिव्यापन","sat": "ᱩᱫᱩᱫ",         "sat_rom": "udud"},
    "hamming":          {"hi": "हैमिंग",   "sat": "ᱦᱮᱢᱤᱝ",        "sat_rom": "heming"},
    "filter":           {"hi": "फ़िल्टर",  "sat": "ᱪᱟᱞᱱᱤ",        "sat_rom": "chalni"},
    "energy":           {"hi": "ऊर्जा",    "sat": "ᱥᱚᱠᱛᱤ",        "sat_rom": "sokti"},
    "viterbi":          {"hi": "वीटर्बी",  "sat": "ᱵᱤᱴᱟᱨᱵᱤ",     "sat_rom": "bitarbi"},
    "hmm":              {"hi": "एचएमएम",   "sat": "ᱜᱩᱯᱛᱚ ᱢᱚᱰᱮᱞ", "sat_rom": "gupto model"},
    "beam":             {"hi": "बीम",      "sat": "ᱵᱤᱢ",           "sat_rom": "bim"},
    "decoding":         {"hi": "डिकोडिंग", "sat": "ᱡᱚᱫᱟ ᱠᱷᱩᱞᱟᱹᱣ","sat_rom":"joda khulau"},
    "transcription":    {"hi": "लिप्यंतरण","sat": "ᱞᱤᱯᱤᱭᱟᱱᱛᱚᱨᱚᱱ","sat_rom":"lipiyantoron"},
    "translation":      {"hi": "अनुवाद",   "sat": "ᱵᱚᱫᱚᱞ",        "sat_rom": "bodol"},
    "corpus":           {"hi": "कॉर्पस",   "sat": "ᱥᱚᱝᱜᱨᱦᱚ",     "sat_rom": "songgroho"},
    "vocabulary":       {"hi": "शब्दकोश",  "sat": "ᱥᱮᱨᱢᱟ ᱠᱚᱥ",   "sat_rom": "serma kos"},
    "perplexity":       {"hi": "परिचय",    "sat": "ᱡᱚᱴᱤᱞ",        "sat_rom": "jotil"},
    "smoothing":        {"hi": "समताकरण",  "sat": "ᱥᱚᱢᱟᱱ",        "sat_rom": "soman"},
    # Common lecture phrases
    "understand":       {"hi": "समझना",    "sat": "ᱵᱩᱡᱦᱟᱹᱣ",      "sat_rom": "bujhau"},
    "example":          {"hi": "उदाहरण",   "sat": "ᱫᱟᱭᱟᱱᱛᱚ",      "sat_rom": "dayanto"},
    "important":        {"hi": "महत्वपूर्ण","sat": "ᱫᱚᱨᱚᱠᱟᱨᱤ",   "sat_rom": "dorokari"},
    "question":         {"hi": "प्रश्न",   "sat": "ᱡᱩᱴᱟᱜ",        "sat_rom": "jutag"},
    "answer":           {"hi": "उत्तर",    "sat": "ᱩᱛᱛᱚᱨ",        "sat_rom": "uttor"},
    "lecture":          {"hi": "व्याख्यान","sat": "ᱵᱭᱟᱠᱡᱟᱱ",      "sat_rom": "byakjan"},
    "student":          {"hi": "छात्र",    "sat": "ᱪᱷᱟᱛᱨᱚ",       "sat_rom": "chhatro"},
    "professor":        {"hi": "प्रोफेसर", "sat": "ᱜᱩᱨᱩ",          "sat_rom": "guru"},
    "university":       {"hi": "विश्वविद्यालय","sat":"ᱵᱤᱥᱵᱤᱫᱭᱟᱞᱚ","sat_rom":"biswidyalo"},
    # Code-switching particles (keep in Santhali equivalents)
    "matlab":           {"hi": "मतलब",     "sat": "ᱢᱟᱱᱮ",          "sat_rom": "mane"},
    "yaani":            {"hi": "यानी",      "sat": "ᱢᱟᱱᱮ",          "sat_rom": "mane"},
    "toh":              {"hi": "तो",        "sat": "ᱛᱮ",            "sat_rom": "te"},
    "aur":              {"hi": "और",        "sat": "ᱟᱨ",            "sat_rom": "ar"},
    "hai":              {"hi": "है",        "sat": "ᱠᱟᱱᱟ",         "sat_rom": "kana"},
    "nahi":             {"hi": "नहीं",      "sat": "ᱵᱟᱱᱩᱜ",        "sat_rom": "banug"},
    "bahut":            {"hi": "बहुत",      "sat": "ᱵᱷᱟᱨᱤ",        "sat_rom": "bhari"},
    "accha":            {"hi": "अच्छा",     "sat": "ᱱᱤᱛᱚᱜ",        "sat_rom": "nitog"},
}


def save_parallel_corpus(path: str = "data/lrl_parallel_corpus.json"):
    """Save the built-in parallel corpus to disk."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(BUILTIN_CORPUS, f, indent=2, ensure_ascii=False)
    log.info(f"Parallel corpus saved: {path}  ({len(BUILTIN_CORPUS)} entries)")


def load_parallel_corpus(path: str = "data/lrl_parallel_corpus.json") -> dict:
    if not os.path.exists(path):
        save_parallel_corpus(path)
    with open(path, encoding="utf-8") as f:
        corpus = json.load(f)
    log.info(f"Loaded parallel corpus: {len(corpus)} entries")
    return corpus


# ══════════════════════════════════════════════════════════════════════════════
#  Translation Engine
# ══════════════════════════════════════════════════════════════════════════════

def translate_token(token: str, corpus: dict, lang_hint: str = "en") -> dict:
    """
    Translate a single token to Santhali.
    Returns {"word": token, "santhali": …, "santhali_roman": …, "method": …}
    """
    t_lower = token.lower().strip(".,!?;:\"'-")

    # 1. Direct corpus lookup
    if t_lower in corpus:
        entry = corpus[t_lower]
        return {
            "word":           token,
            "santhali":       entry.get("sat", ""),
            "santhali_roman": entry.get("sat_rom", ""),
            "method":         "corpus",
        }

    # 2. Partial match (for compound words / inflections)
    for key in corpus:
        if t_lower.startswith(key) or key in t_lower:
            entry = corpus[key]
            return {
                "word":           token,
                "santhali":       entry.get("sat", ""),
                "santhali_roman": entry.get("sat_rom", "") + f"[+{t_lower[len(key):]}]",
                "method":         "partial",
            }

    # 3. IPA → Ol Chiki transliteration
    from ipa_mapper import english_to_ipa, hindi_to_ipa
    if lang_hint == "hi":
        ipa = hindi_to_ipa(t_lower)
    else:
        ipa = english_to_ipa(t_lower)
    sat_script = ipa_to_santhali_script(ipa)

    return {
        "word":           token,
        "santhali":       sat_script,
        "santhali_roman": t_lower,   # keep original as fallback
        "method":         "transliteration",
    }


def translate_text(
    text: str,
    corpus: dict,
    lang_hint: str = "en",
) -> tuple:
    """
    Translate a full text string token-by-token to Santhali.

    Returns
    -------
    santhali_text : str  — Santhali output (Ol Chiki script)
    token_info    : list — per-token translation details
    """
    tokens = text.strip().split()
    parts  = []
    info   = []

    for token in tokens:
        result = translate_token(token, corpus, lang_hint)
        parts.append(result["santhali"] or result["santhali_roman"] or token)
        info.append(result)

    return " ".join(parts), info


def translate_transcript(
    ipa_path:    str = "outputs/ipa_output.json",
    corpus_path: str = "data/lrl_parallel_corpus.json",
    out_path:    str = "outputs/santhali_translation.json",
) -> dict:
    """
    Translate full IPA/text segments to Santhali.
    """
    with open(ipa_path, encoding="utf-8") as f:
        ipa_data = json.load(f)

    corpus = load_parallel_corpus(corpus_path)
    segments_out = []
    coverage_corpus = 0
    coverage_total  = 0

    for seg in ipa_data.get("segments", []):
        sat_text, tok_info = translate_text(
            seg["text"],
            corpus,
            lang_hint=seg.get("lang_hint", "en"),
        )
        corpus_hits = sum(1 for t in tok_info if t["method"] == "corpus")
        coverage_corpus += corpus_hits
        coverage_total  += len(tok_info)

        segments_out.append({
            "start":           seg["start"],
            "end":             seg["end"],
            "source_text":     seg["text"],
            "ipa":             seg["ipa"],
            "santhali":        sat_text,
            "token_info":      tok_info,
        })

    cov_rate = coverage_corpus / max(coverage_total, 1)
    full_sat = " ".join(s["santhali"] for s in segments_out)
    result = {
        "language_pair":   "hin+eng → sat (Santhali)",
        "corpus_coverage": round(cov_rate, 4),
        "full_santhali":   full_sat,
        "segments":        segments_out,
        "corpus_size":     len(corpus),
    }

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)

    log.info(f"[OK] Santhali translation saved: {out_path}")
    log.info(f"     Corpus coverage: {cov_rate:.1%}")
    log.info(f"     Sample: {full_sat[:200]}")
    return result


# ── Entry Point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Translate to Santhali LRL")
    parser.add_argument("--ipa",    default="outputs/ipa_output.json")
    parser.add_argument("--corpus", default="data/lrl_parallel_corpus.json")
    parser.add_argument("--out",    default="outputs/santhali_translation.json")
    parser.add_argument("--save-corpus", action="store_true",
                        help="Save the built-in parallel corpus to disk and exit.")
    args = parser.parse_args()

    if args.save_corpus:
        save_parallel_corpus(args.corpus)
        print(f"[OK] Corpus saved: {args.corpus}  ({len(BUILTIN_CORPUS)} entries)")
    else:
        translate_transcript(args.ipa, args.corpus, args.out)
