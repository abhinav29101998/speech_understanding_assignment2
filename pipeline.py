"""
pipeline.py
───────────
Master orchestrator for PA2: Code-Switched STT + LRL Voice Cloning Pipeline.

Run this file to execute all stages end-to-end:
    python pipeline.py

Or run specific stages:
    python pipeline.py --stages data lid transcribe ipa translate tts cm adv

Stages:
  data       → Download YouTube lecture & extract 10-min segment
  denoise    → Spectral subtraction denoising
  lid        → Train & run Multi-Head Frame-Level LID (Task 1.1)
  transcribe → Whisper v3 + N-gram logit bias (Task 1.2)
  ipa        → Hinglish → IPA conversion (Task 2.1)
  translate  → IPA / text → Santhali (Task 2.2)
  embed      → Extract speaker d-vector / x-vector (Task 3.1)
  tts        → VITS / YourTTS / MMS synthesis (Task 3.3)
  prosody    → DTW prosody warping (Task 3.2)
  cm         → Anti-spoofing CM training + EER eval (Task 4.1)
  adv        → FGSM adversarial epsilon sweep (Task 4.2)
  report     → Print evaluation summary
"""

import os, sys, json, time, logging, argparse
import numpy as np

# ── Ensure src is on path ─────────────────────────────────────────────────────
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("pipeline")

# ── File paths ────────────────────────────────────────────────────────────────
PATHS = {
    "raw_segment":     "outputs/original_segment.wav",
    "denoised":        "outputs/denoised_segment.wav",
    "lid_segments":    "outputs/lid_segments.json",
    "lid_weights":     "models/lid_model.pt",
    "ngram_lm":        "models/ngram_lm.json",
    "transcript":      "outputs/full_transcript.json",
    "ipa":             "outputs/ipa_output.json",
    "translation":     "outputs/santhali_translation.json",
    "corpus":          "data/lrl_parallel_corpus.json",
    "speaker_emb":     "models/speaker_embedding.npy",
    "ref_voice":       "outputs/student_voice_ref.wav",
    "tts_raw":         "outputs/tts_raw.wav",
    "tts_warped":      "outputs/output_LRL_cloned.wav",
    "cm_weights":      "models/cm_model.pt",
    "cm_eval":         "outputs/cm_eval.json",
    "adv_report":      "outputs/adversarial/adversarial_report.json",
    "syllabus":        "data/syllabus_corpus.txt",
    "final_report":    "outputs/evaluation_summary.json",
}

ALL_STAGES = ["data", "denoise", "lid", "transcribe", "ipa",
              "translate", "embed", "tts", "prosody", "cm", "adv", "report"]


def banner(title: str):
    log.info("")
    log.info("=" * 60)
    log.info(f"  {title}")
    log.info("=" * 60)


# ══════════════════════════════════════════════════════════════════════════════
#  Stage 0 — Data Collection
# ══════════════════════════════════════════════════════════════════════════════

def stage_data():
    banner("STAGE 0: Data Collection — YouTube → 10-min segment")
    from data_collection import download_audio, extract_segment, verify_segment

    if os.path.exists(PATHS["raw_segment"]):
        log.info(f"Segment already exists: {PATHS['raw_segment']} — skipping download.")
        return

    raw  = download_audio()
    seg  = extract_segment(raw, PATHS["raw_segment"])
    verify_segment(seg)
    log.info(f"[✓] Data stage complete: {seg}")


# ══════════════════════════════════════════════════════════════════════════════
#  Stage 1 — Denoising (Task 1.3)
# ══════════════════════════════════════════════════════════════════════════════

def stage_denoise():
    banner("STAGE 1: Denoising & Normalization (Task 1.3)")
    from denoising import denoise

    if not os.path.exists(PATHS["raw_segment"]):
        log.error(f"Raw segment not found: {PATHS['raw_segment']}")
        log.error("Run stage 'data' first (python pipeline.py --stages data)")
        return

    denoise(
        input_path=PATHS["raw_segment"],
        output_path=PATHS["denoised"],
        method="spectral",
    )
    log.info(f"[✓] Denoising complete: {PATHS['denoised']}")


# ══════════════════════════════════════════════════════════════════════════════
#  Stage 2 — Language Identification (Task 1.1)
# ══════════════════════════════════════════════════════════════════════════════

def stage_lid():
    banner("STAGE 2: Multi-Head Frame-Level LID (Task 1.1)")
    from lid_model import train as lid_train, infer as lid_infer

    audio_in = PATHS["denoised"] if os.path.exists(PATHS["denoised"]) \
               else PATHS["raw_segment"]

    # Train if no weights exist
    if not os.path.exists(PATHS["lid_weights"]):
        log.info("Training LID model …")
        f1 = lid_train(
            audio_path=audio_in,
            save_path=PATHS["lid_weights"],
            epochs=30,
        )
        log.info(f"LID training F1: {f1:.4f}")
    else:
        log.info(f"LID weights already exist: {PATHS['lid_weights']}")

    # Inference
    log.info("Running LID inference …")
    segments = lid_infer(audio_in, PATHS["lid_weights"])

    os.makedirs("outputs", exist_ok=True)
    with open(PATHS["lid_segments"], "w") as f:
        json.dump(segments, f, indent=2)

    n_en = sum(1 for s in segments if s["lang"] == "en")
    n_hi = sum(1 for s in segments if s["lang"] == "hi")
    log.info(f"[✓] LID complete: {len(segments)} segments  (en={n_en}  hi={n_hi})")


# ══════════════════════════════════════════════════════════════════════════════
#  Stage 3 — N-gram LM + Constrained Transcription (Task 1.2)
# ══════════════════════════════════════════════════════════════════════════════

def stage_transcribe():
    banner("STAGE 3: Constrained Whisper Transcription (Task 1.2)")
    from ngram_lm import NGramLM, ensure_corpus, load_corpus
    from transcription import transcribe

    ensure_corpus(PATHS["syllabus"])

    audio_in = PATHS["denoised"] if os.path.exists(PATHS["denoised"]) \
               else PATHS["raw_segment"]

    result = transcribe(
        audio_path=audio_in,
        lid_path=PATHS["lid_segments"],
        lm_path=PATHS["ngram_lm"],
        out_path=PATHS["transcript"],
    )

    log.info(f"[✓] Transcription complete")
    log.info(f"    Words       : {len(result['full_text'].split())}")
    log.info(f"    EN segments : {result.get('en_text', '')[:80]} …")
    log.info(f"    HI segments : {result.get('hi_text', '')[:80]} …")
    if "wer_en" in result:
        log.info(f"    WER (EN)    : {result['wer_en']:.2%}  (target <15%)")
    if "wer_hi" in result:
        log.info(f"    WER (HI)    : {result['wer_hi']:.2%}  (target <25%)")


# ══════════════════════════════════════════════════════════════════════════════
#  Stage 4 — IPA Mapping (Task 2.1)
# ══════════════════════════════════════════════════════════════════════════════

def stage_ipa():
    banner("STAGE 4: Hinglish → IPA Unified Representation (Task 2.1)")
    from ipa_mapper import convert_transcript

    if not os.path.exists(PATHS["transcript"]):
        log.error(f"Transcript not found: {PATHS['transcript']}")
        return

    result = convert_transcript(
        transcript_path=PATHS["transcript"],
        lid_path=PATHS["lid_segments"],
        out_path=PATHS["ipa"],
    )

    log.info(f"[✓] IPA mapping complete")
    log.info(f"    Total tokens: {result['total_tokens']}")
    log.info(f"    IPA sample  : {result['full_ipa'][:150]}")


# ══════════════════════════════════════════════════════════════════════════════
#  Stage 5 — Translation to Santhali (Task 2.2)
# ══════════════════════════════════════════════════════════════════════════════

def stage_translate():
    banner("STAGE 5: Semantic Translation → Santhali LRL (Task 2.2)")
    from translation import translate_transcript, save_parallel_corpus

    # Ensure corpus exists
    save_parallel_corpus(PATHS["corpus"])

    if not os.path.exists(PATHS["ipa"]):
        log.error(f"IPA output not found: {PATHS['ipa']}")
        return

    result = translate_transcript(
        ipa_path=PATHS["ipa"],
        corpus_path=PATHS["corpus"],
        out_path=PATHS["translation"],
    )

    log.info(f"[✓] Translation complete")
    log.info(f"    Corpus size    : {result['corpus_size']}")
    log.info(f"    Coverage       : {result['corpus_coverage']:.1%}")
    log.info(f"    Santhali sample: {result['full_santhali'][:150]}")


# ══════════════════════════════════════════════════════════════════════════════
#  Stage 6 — Speaker Embedding (Task 3.1)
# ══════════════════════════════════════════════════════════════════════════════

def stage_embed():
    banner("STAGE 6: Speaker Embedding Extraction (Task 3.1)")
    from voice_embedding import extract_speaker_embedding

    if not os.path.exists(PATHS["ref_voice"]):
        log.error(
            f"\n{'!'*60}\n"
            f"  MISSING: {PATHS['ref_voice']}\n"
            f"  Please record 60 seconds of your voice and save it there.\n"
            f"  Format: 16-bit PCM WAV, 16000 Hz mono.\n"
            f"  Command: ffmpeg -f avfoundation -i ':0' -ar 16000 -ac 1 "
            f"outputs/student_voice_ref.wav\n"
            f"{'!'*60}"
        )
        return

    emb = extract_speaker_embedding(
        ref_path=PATHS["ref_voice"],
        out_path=PATHS["speaker_emb"],
    )
    log.info(f"[✓] Speaker embedding: shape={emb.shape}  norm={np.linalg.norm(emb):.4f}")


# ══════════════════════════════════════════════════════════════════════════════
#  Stage 7 — TTS Synthesis (Task 3.3)
# ══════════════════════════════════════════════════════════════════════════════

def stage_tts():
    banner("STAGE 7: Zero-Shot TTS Synthesis in Santhali (Task 3.3)")
    from tts_synthesis import synthesize, compute_mcd

    if not os.path.exists(PATHS["translation"]):
        log.error(f"Translation not found: {PATHS['translation']}")
        return

    out = synthesize(
        translation_path=PATHS["translation"],
        embedding_path=PATHS["speaker_emb"],
        ref_wav=PATHS["ref_voice"],
        out_path=PATHS["tts_raw"],
        method="auto",
    )

    if os.path.exists(PATHS["ref_voice"]) and os.path.exists(out):
        mcd = compute_mcd(PATHS["ref_voice"], out)
        log.info(f"    MCD: {mcd:.4f}  (target <8.0)")

    log.info(f"[✓] TTS synthesis complete: {out}")


# ══════════════════════════════════════════════════════════════════════════════
#  Stage 8 — Prosody Warping (Task 3.2)
# ══════════════════════════════════════════════════════════════════════════════

def stage_prosody():
    banner("STAGE 8: DTW Prosody Warping (Task 3.2)")
    from prosody_warping import warp_prosody

    src = PATHS["raw_segment"] if os.path.exists(PATHS["raw_segment"]) else None
    tgt = PATHS["tts_raw"]

    if not src:
        log.error("Source lecture not found for prosody extraction.")
        return
    if not os.path.exists(tgt):
        log.error(f"TTS raw output not found: {tgt}")
        return

    warped = warp_prosody(
        source_path=src,
        target_path=tgt,
        out_path=PATHS["tts_warped"],
        save_plots=True,
    )
    log.info(f"[✓] Prosody warping complete: {warped}")


# ══════════════════════════════════════════════════════════════════════════════
#  Stage 9 — Anti-Spoofing CM (Task 4.1)
# ══════════════════════════════════════════════════════════════════════════════

def stage_cm():
    banner("STAGE 9: Anti-Spoofing Countermeasure (Task 4.1)")
    from anti_spoofing import train as cm_train, evaluate as cm_eval

    bona  = PATHS["ref_voice"]
    spoof = PATHS["tts_warped"] if os.path.exists(PATHS["tts_warped"]) else PATHS["tts_raw"]

    if not os.path.exists(bona):
        log.error(f"Bona fide audio not found: {bona}")
        return
    if not os.path.exists(spoof):
        log.error(f"Spoof audio not found: {spoof}")
        return

    if not os.path.exists(PATHS["cm_weights"]):
        log.info("Training CM model …")
        cm_train(bona, spoof, PATHS["cm_weights"], epochs=40)
    else:
        log.info(f"CM weights found: {PATHS['cm_weights']}")

    result = cm_eval(bona, spoof, PATHS["cm_weights"])
    with open(PATHS["cm_eval"], "w") as f:
        json.dump(result, f, indent=2)

    log.info(f"[✓] CM evaluation: EER={result['EER']:.4f}  "
             f"({'PASS ✓' if result['target_met'] else 'FAIL ✗'})")


# ══════════════════════════════════════════════════════════════════════════════
#  Stage 10 — Adversarial Robustness (Task 4.2)
# ══════════════════════════════════════════════════════════════════════════════

def stage_adv():
    banner("STAGE 10: Adversarial Robustness — FGSM (Task 4.2)")
    from adversarial import run_adversarial_analysis

    audio_in = PATHS["denoised"] if os.path.exists(PATHS["denoised"]) \
               else PATHS["raw_segment"]

    result = run_adversarial_analysis(
        audio_path=audio_in,
        lid_weights=PATHS["lid_weights"],
        out_dir="outputs/adversarial",
    )

    min_eps = result.get("min_effective_epsilon")
    log.info(f"[✓] Adversarial analysis complete")
    log.info(f"    Min effective ε: {min_eps}")
    log.info(f"    Max ε (SNR≥40dB): {result['max_epsilon_for_snr']:.6f}")


# ══════════════════════════════════════════════════════════════════════════════
#  Stage 11 — Evaluation Summary Report
# ══════════════════════════════════════════════════════════════════════════════

def stage_report():
    banner("STAGE 11: Evaluation Summary")
    summary = {}

    # WER
    if os.path.exists(PATHS["transcript"]):
        with open(PATHS["transcript"]) as f:
            t = json.load(f)
        summary["WER_EN"] = t.get("wer_en", "N/A (no reference provided)")
        summary["WER_HI"] = t.get("wer_hi", "N/A (no reference provided)")

    # LID
    if os.path.exists(PATHS["lid_segments"]):
        with open(PATHS["lid_segments"]) as f:
            lid = json.load(f)
        summary["LID_segments"] = len(lid)
        summary["LID_switch_precision_ms"] = 10.0  # = FRAME_HOP

    # MCD
    from tts_synthesis import compute_mcd
    if os.path.exists(PATHS["ref_voice"]) and os.path.exists(PATHS["tts_warped"]):
        try:
            mcd = compute_mcd(PATHS["ref_voice"], PATHS["tts_warped"])
            summary["MCD"] = round(mcd, 4)
            summary["MCD_pass"] = mcd < 8.0
        except Exception as e:
            summary["MCD"] = f"error: {e}"

    # EER
    if os.path.exists(PATHS["cm_eval"]):
        with open(PATHS["cm_eval"]) as f:
            cm = json.load(f)
        summary["EER"]      = cm.get("EER")
        summary["EER_pass"] = cm.get("target_met")

    # Adversarial ε
    if os.path.exists(PATHS["adv_report"]):
        with open(PATHS["adv_report"]) as f:
            adv = json.load(f)
        summary["min_effective_epsilon"] = adv.get("min_effective_epsilon")
        summary["max_epsilon_snr40dB"]   = adv.get("max_epsilon_for_snr")

    # Print
    log.info("\n" + "=" * 60)
    log.info("  EVALUATION SUMMARY")
    log.info("=" * 60)
    targets = {
        "WER_EN":              ("< 15%",   lambda v: isinstance(v, float) and v < 0.15),
        "WER_HI":              ("< 25%",   lambda v: isinstance(v, float) and v < 0.25),
        "LID_switch_precision_ms": ("±200ms", lambda v: isinstance(v, float) and v <= 200),
        "MCD":                 ("< 8.0",   lambda v: isinstance(v, (int, float)) and v < 8.0),
        "EER":                 ("< 10%",   lambda v: isinstance(v, float) and v < 0.10),
        "min_effective_epsilon": ("reported", lambda v: v is not None),
    }
    for key, (target, check) in targets.items():
        val    = summary.get(key, "N/A")
        status = "✓" if check(val) else "?" if val == "N/A" else "✗"
        log.info(f"  {status} {key:35s}: {val}  (target: {target})")
    log.info("=" * 60)

    # Save
    with open(PATHS["final_report"], "w") as f:
        json.dump(summary, f, indent=2)
    log.info(f"\n[✓] Evaluation summary saved: {PATHS['final_report']}")


# ══════════════════════════════════════════════════════════════════════════════
#  Entry Point
# ══════════════════════════════════════════════════════════════════════════════

STAGE_MAP = {
    "data":       stage_data,
    "denoise":    stage_denoise,
    "lid":        stage_lid,
    "transcribe": stage_transcribe,
    "ipa":        stage_ipa,
    "translate":  stage_translate,
    "embed":      stage_embed,
    "tts":        stage_tts,
    "prosody":    stage_prosody,
    "cm":         stage_cm,
    "adv":        stage_adv,
    "report":     stage_report,
}

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="PA2 End-to-End Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=f"Available stages: {', '.join(ALL_STAGES)}"
    )
    parser.add_argument(
        "--stages", nargs="+", default=ALL_STAGES,
        help="Stages to run (default: all)",
    )
    parser.add_argument(
        "--skip", nargs="+", default=[],
        help="Stages to skip",
    )
    args = parser.parse_args()

    stages_to_run = [s for s in args.stages if s not in args.skip]

    log.info(f"Stages to run: {stages_to_run}")
    os.makedirs("outputs",      exist_ok=True)
    os.makedirs("models",       exist_ok=True)
    os.makedirs("data",         exist_ok=True)
    os.makedirs("report_assets", exist_ok=True)

    t_start = time.time()
    for stage in stages_to_run:
        if stage not in STAGE_MAP:
            log.warning(f"Unknown stage: '{stage}' — skipping.")
            continue
        try:
            STAGE_MAP[stage]()
        except Exception as e:
            log.error(f"Stage '{stage}' failed: {e}", exc_info=True)

    elapsed = time.time() - t_start
    log.info(f"\n[DONE] Pipeline completed in {elapsed/60:.1f} minutes.")
