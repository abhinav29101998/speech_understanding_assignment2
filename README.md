# Speech Understanding — Programming Assignment 2
## Code-Switched Lecture Transcription & LRL Voice Cloning Pipeline

### Repository Structure
```
PA2/
├── README.md
├── requirements.txt
├── pipeline.py                  # Master orchestrator — run this end-to-end
├── src/
│   ├── data_collection.py       # YouTube scraping + segment extraction
│   ├── denoising.py             # Spectral Subtraction denoiser
│   ├── lid_model.py             # Multi-Head Frame-Level LID (Task 1.1)
│   ├── ngram_lm.py              # N-gram LM trained on syllabus (Task 1.2)
│   ├── constrained_decoding.py  # Logit-bias + constrained beam search (Task 1.2)
│   ├── transcription.py         # Whisper-v3 wrapper with constrained decoding
│   ├── ipa_mapper.py            # Hinglish → IPA phonetic mapping (Task 2.1)
│   ├── translation.py           # Semantic translation to LRL / parallel corpus (Task 2.2)
│   ├── voice_embedding.py       # d-vector / x-vector extraction (Task 3.1)
│   ├── prosody_warping.py       # F0 + Energy DTW warping (Task 3.2)
│   ├── tts_synthesis.py         # VITS / YourTTS zero-shot synthesis (Task 3.3)
│   ├── anti_spoofing.py         # LFCC-based CM classifier + EER (Task 4.1)
│   └── adversarial.py           # FGSM adversarial noise injection (Task 4.2)
├── data/
│   ├── syllabus_corpus.txt      # Speech course syllabus for N-gram LM
│   ├── hinglish_ipa_map.json    # Custom Hinglish IPA phoneme table
│   └── lrl_parallel_corpus.json # 500-word Santhali technical parallel corpus
├── models/                      # Downloaded/saved model weights (git-ignored)
└── outputs/
    ├── original_segment.wav
    ├── student_voice_ref.wav    # Place your 60s recording here before running
    └── output_LRL_cloned.wav
```

---

## Setup

```bash
# 1. Create virtual environment
python3 -m venv venv && source venv/bin/activate

# 2. Install all dependencies
pip install -r requirements.txt

# 3. Record your voice (60 seconds, 22050 Hz) and place at:
#    outputs/student_voice_ref.wav

# 4. Run full pipeline end-to-end
python pipeline.py

# Or run individual stages:
python src/data_collection.py          # Download & trim YouTube segment
python src/lid_model.py --train        # Train / evaluate LID
python src/anti_spoofing.py --eval     # Evaluate anti-spoofing EER
python src/adversarial.py              # Run FGSM attack
```

---

## Target LRL: Santhali (ISO 639-3: sat)

Chosen because:
- Officially recognized in India's 8th Schedule (~7M speakers).
- No large-scale MT model exists → requires a custom parallel corpus (Task 2.2).
- Unique phonology (aspirated stops, retroflex) makes IPA mapping non-trivial.

---

## Evaluation Benchmarks

| Metric | Requirement |
|--------|-------------|
| WER (English) | < 15% |
| WER (Hindi) | < 25% |
| MCD | < 8.0 |
| LID Switch Precision | ±200 ms |
| Anti-Spoofing EER | < 10% |
| Adversarial ε | reported |

---

## References
- Radford et al. (2022). Whisper. OpenAI.
- Baevski et al. (2020). wav2vec 2.0. NeurIPS.
- Kim et al. (2021). VITS. ICML.
- Todisco et al. (2019). ASSERT Anti-Spoofing.
- Goodfellow et al. (2014). FGSM. ICLR.
