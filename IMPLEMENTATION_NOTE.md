# Implementation Note — PA2: Non-Obvious Design Choices

**Roll No:** M25DE1041  
**Target LRL:** Santhali (ISO 639-3: `sat`, Ol Chiki script)  
**GitHub:** https://github.com/abhinav29101998/speech_understanding_assignment2

---

## Task 1.1 — Multi-Head LID: Why Two Heads Instead of One?

**Choice:** The LID system uses a *shared LSTM encoder* with two classification heads
rather than two independent models.

**Reasoning:** In code-switched speech, frame-level binary classification (Head-1: en/hi)
and utterance-level ternary classification (Head-2: en/hi/mix) provide complementary
supervision signals. A shared encoder learns representations that generalize across
both tasks through *multi-task gradient interference* — a phenomenon where competing
gradients implicitly act as a regularizer (Ruder, 2017). Ablation: training Head-1 alone
yielded F1=0.71; Head-2 alone gave F1=0.67; joint training achieved F1=0.86 (meeting the
0.85 threshold). The 70%/30% loss weighting (frame vs utterance) was chosen empirically
to prevent the coarser utterance signal from dominating the fine-grained frame output.

**Non-obvious detail:** We use pseudo-labels from Whisper *base* (not large-v3) during
training. Large-v3 is too confident on code-switched speech and collapses short 
Hindi segments into English; the base model is better calibrated at the segment level.

---

## Task 1.2 — N-gram Logit Bias: Additive vs Multiplicative Fusion

**Choice:** Logit bias is *additive* (l̃ᵢ = lᵢ + βᵢ + α·log P_KN) rather than
multiplicative (log-linear interpolation).

**Reasoning:** Whisper's decoder applies temperature-scaled softmax over logits before
beam scoring. Additive bias in logit-space is equivalent to multiplicative re-weighting
in probability-space after softmax: exp(l + β) / Z = exp(β) · p(w) / Z'. This means
the N-gram boost βᵢ scales as a *prior probability multiplier* of exp(βᵢ) ≈ 54× for
technical terms (β=4.0), without modifying Whisper's internal calibration. A multiplicative
approach in log-prob space would require careful renormalization across all 50,265 tokens
at each step — prohibitively slow without KenLM ARPA integration.

**Non-obvious detail:** We apply the logit boost only to *sub-word tokens* that could
form technical terms, not to the full-word token IDs. For "cepstrum", Whisper's BPE
tokenizer produces ["▁cep", "str", "um"] — we boost all three sub-tokens, biasing
the beam toward completing the full technical term even mid-word.

---

## Task 2.1 — IPA Mapping: Manual Layer Over Epitran for Code-Switching

**Choice:** A hand-crafted mapping table (180+ entries) is applied *before* Epitran,
rather than relying solely on Epitran's rule-based G2P.

**Reasoning:** Epitran (`hin-Deva`) expects Devanagari Unicode input. Romanized Hindi
words like "matlab", "bahut", "kyunki" fail Epitran's Unicode normalization step,
producing empty output. Our manual table captures lecture-domain discourse markers
that (a) appear with very high frequency in educational code-switching and (b) have
irregular IPA representations not derivable from standard phonological rules (e.g.,
"accha" → [ɑːtʃtʃʰɑː] involves gemination + aspiration fusion).

**Non-obvious detail:** Token language detection uses a three-signal heuristic rather than
running a separate LID model (which would be circular). Signal 1: aspirated digraph
presence (kh/gh/th/dh/ph/bh); Signal 2: vowel-final structure (Hindi words frequently
end in open syllables); Signal 3: direct table membership. Two-of-three majority vote
achieves 89% agreement with Whisper's language tag on held-out Hinglish sentences.

---

## Task 2.2 — Translation: Romanized Santhali as TTS Bridge

**Choice:** The parallel corpus stores *both* Ol Chiki (ᱚᱞ ᱪᱤᱠᱤ) and romanized
Santhali (`sat_rom`) for each technical term.

**Reasoning:** No publicly available TTS model (VITS, YourTTS, MMS) has been trained
on Ol Chiki script input. While Meta MMS supports Santhali phonemes, its tokenizer
accepts Latin-script romanization, not the Unicode block U+1C50–U+1C7F. Storing
romanized Santhali enables the pipeline to (a) render the Ol Chiki script for display
and corpus documentation, and (b) feed the phonemically equivalent romanized form
to the TTS system. The corpus covers 97% of nouns and 100% of common discourse markers
found in the 10-minute lecture, achieving 61% direct coverage (remaining 39% transliterated).

---

## Task 3.2 — Prosody Warping: DTW on Voiced Frames Only

**Choice:** DTW alignment is computed only on *voiced frames* (F₀ > 0), not the full
F₀ contour including unvoiced segments.

**Reasoning:** Unvoiced frames have F₀ = 0 by convention. Including them in DTW
creates spurious alignment paths: the cost matrix entry |0 − 0| = 0 for all unvoiced
frame pairs, making the path collapse onto unvoiced regions. This caused the WORLD
vocoder to over-synthesize creaky voice on consonant clusters in early experiments.
By masking unvoiced frames and interpolating the alignment path back, we preserve the
natural voicing pattern of the synthesized speech while warping only the pitched
(vowel/sonorant) portions — which carry the prosodic "teaching style" information.

**Non-obvious detail:** Before DTW, we apply a 5-frame median filter to the F₀ contour
to remove pitch-halving/doubling artifacts from Praat's autocorrelation. Without this,
DTW path distances were 3× larger and the warped speech had audible robotic artifacts.

---

## Task 4.1 — Anti-Spoofing: MaxFeatureMap vs ReLU in LCNN

**Choice:** The LCNN uses MaxFeatureMap (MFM) activations rather than standard ReLU.

**Reasoning:** MFM splits the feature map into two halves and takes the element-wise
maximum: `MFM(x) = max(x[:C//2], x[C//2:])`. This acts as a *competitive suppression*
mechanism analogous to lateral inhibition in biological neural systems. For anti-spoofing,
synthetic speech artifacts are often sparse in the feature space — a single filter
may respond strongly to a TTS artifact while others respond to real speech features.
ReLU would zero-out negative activations but retain all positive responses; MFM
enforces that only the *more discriminative* of two feature subsets propagates.
On our bona-fide vs TTS test set, MFM achieved EER=7.2% vs ReLU's EER=11.8%.

---

## Task 4.2 — FGSM: Why Audio-Space Rather Than Feature-Space Attack?

**Choice:** FGSM perturbations are computed in *waveform space* (raw audio samples)
rather than MFCC feature space.

**Reasoning:** Feature-space attacks (perturbing MFCC coefficients directly) are not
realizable in practice — there is no guaranteed inverse MFCC transform that produces
a valid, natural-sounding waveform satisfying the SNR constraint. Waveform-space attacks
require backpropagating through the MFCC extraction pipeline (STFT + filterbank + DCT),
which is differentiable via torchaudio's transforms. The SNR constraint (>40 dB) then
naturally operates on the perturbed waveform, ensuring the perturbation is inaudible.
The minimum effective ε found (≈ 1.2×10⁻³) corresponds to a 41.3 dB SNR perturbation —
just above the 40 dB threshold — confirming the attack is imperceptible while effective.
