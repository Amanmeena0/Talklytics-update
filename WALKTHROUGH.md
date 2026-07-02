# Talklytics — Project Walkthrough

> **Talklytics** is a real-time multimodal speech analysis system that
> detects conversational interest and engagement during live sales
> conversations, providing actionable feedback *while the conversation is
> still happening* — not after.

---

## 1  System Overview

Talklytics follows a modular pipeline architecture inspired by
production-grade real-time analytics systems:

```
Microphone → Audio Capture → Preprocessing → ┬─ Acoustic Features (MFCC, Energy, Pitch, ZCR)
                                              │
                                              ├─ ASR Transcription (Whisper)
                                              │       │
                                              │       └─ Linguistic Analysis
                                              │             ├─ Sentiment (DistilBERT)
                                              │             ├─ Keyword Detection
                                              │             └─ Intent Classification
                                              │
                                              └─ Feature-Level Fusion → Engagement Score → Dashboard
```

Each stage is implemented as an independent, testable module:

| Module | Responsibility |
|--------|---------------|
| `audio_capture.py` | Captures live audio via PyAudio in 3-second segments |
| `audio_preprocessor.py` | Silence gating, normalisation, noise reduction |
| `acoustic_extractor.py` | Extracts MFCC, energy, pitch, zero-crossing rate |
| `speech_recognizer.py` | Whisper-based ASR transcription |
| `linguistic_analyzer.py` | Sentiment analysis, keyword detection, intent classification |
| `fusion_model.py` | Feature-level fusion using Random Forest classifier |
| `engagement_tracker.py` | Time-series engagement tracking |
| `pipeline.py` | Orchestrates end-to-end processing in a background thread |
| `dashboard/app.py` | Real-time Streamlit dashboard with 3-zone layout |

---

## 2  Feature Engineering

### 2.1  Acoustic Features

Four core features are extracted from each 3-second audio segment:

- **MFCC** (13 coefficients + means) — Captures the spectral envelope of
  speech, the standard representation for speech/emotion recognition.
- **Energy** (RMS) — Measures vocal intensity; high energy correlates
  with enthusiasm and engagement.
- **Pitch** (fundamental frequency via autocorrelation) — Pitch
  variation signals emotional expressiveness; monotone delivery
  suggests disengagement.
- **Zero-Crossing Rate** (ZCR) — Distinguishes voiced from unvoiced
  speech; useful for detecting hesitation patterns.

### 2.2  Linguistic Features

- **Sentiment** — DistilBERT fine-tuned on SST-2 provides
  POSITIVE/NEGATIVE/NEUTRAL labels with confidence scores.
- **Buying-Signal Keywords** — Domain-specific keyword matching
  (e.g. "pricing", "next steps", "demo").
- **Hesitation Indicators** — Phrases that signal reluctance
  (e.g. "not sure", "maybe later", "too expensive").

### 2.3  Intent Detection (Smart NLP Feature)

A rule-based intent classifier detects five sales-conversation intents:

| Intent | Purpose | Example Triggers |
|--------|---------|-----------------|
| `PRICING` | Customer is asking about cost | "how much", "budget", "per month" |
| `COMPARISON` | Customer is evaluating alternatives | "compared to", "better than", "other options" |
| `OBJECTION` | Customer is raising concerns | "too expensive", "not convinced", "won't work" |
| `COMMITMENT` | Customer is ready to proceed | "let's do it", "next steps", "sign me up" |
| `INFORMATION` | Customer wants to learn more | "tell me more", "how does it work", "demo" |

**Design rationale:** A rule-based approach was chosen over a
trained classifier because:
1. It requires no additional training data.
2. It is fully interpretable — every classification can be
   traced to a specific trigger phrase.
3. It operates with zero latency overhead.
4. It can be extended simply by adding keywords to `config.py`.

---

## 3  Fusion Strategy

Talklytics uses **feature-level (early) fusion**:

1. Acoustic features are encoded as a fixed-length numeric vector
   (13 MFCC means + energy + pitch + ZCR = 16 values).
2. Linguistic features are encoded as a 6-value vector:
   `[sentiment_encoded, sentiment_score, buying_count, hesitation_count, intent_count, intent_confidence]`.
3. Both vectors are **concatenated** into a single 22-dimensional
   feature vector.
4. A Random Forest classifier maps this fused vector to a 1–5
   engagement score.

**Why feature-level fusion?**
- It preserves all modality-specific information for the classifier.
- Random Forest handles heterogeneous feature types
  (continuous + categorical) naturally.
- It avoids the complexity of decision-level fusion (separate models
  per modality) while still capturing cross-modal interactions.

**Future work:** Cross-modal attention mechanisms (e.g. a
Transformer-based fusion layer) could learn more nuanced
interactions between acoustic and linguistic cues, particularly
for detecting sarcasm or feigned enthusiasm.

---

## 4  Synthetic Model Limitation

> *"The current model is trained on synthetic data for pipeline
> validation. The system is designed to be extended using real-world
> datasets such as RAVDESS and MELD for improved generalization."*

The synthetic training script (`training/train_from_synthetic.py`)
generates labeled feature vectors with controlled distributions
to validate the full pipeline — from feature extraction through
model inference to dashboard display. This approach:

- Verifies that all modules integrate correctly end-to-end.
- Provides deterministic, reproducible baselines for development.
- Enables rapid iteration without the overhead of dataset
  collection and annotation.

The architecture is deliberately designed for dataset-agnostic
extensibility: swapping in real-world data requires only
modifying the training script while the entire inference pipeline
remains unchanged.

---

## 5  Accuracy Disclaimer

> *"The synthetic dataset helps validate the pipeline. Real-world
> accuracy would be lower and is addressed using external datasets."*

Reported accuracy metrics (from synthetic training) reflect the
model's ability to learn the generated feature distributions — not
real-world conversational patterns. The primary validation goal is
**pipeline correctness**, not classification accuracy. Real-world
deployment would require:

1. Collection of annotated sales conversation data.
2. Re-training with datasets like RAVDESS (emotional speech) or
   MELD (multimodal dialogue).
3. Cross-validation with held-out real conversation segments.

---

## 6  Real-Time Simulation

Talklytics achieves live feedback using an engineering workaround
for Streamlit's single-threaded architecture:

1. **Background thread** — The pipeline runs in a daemon thread,
   continuously processing audio segments and pushing results to a
   thread-safe `queue.Queue`.
2. **Polling via `st.rerun()`** — The dashboard re-renders every
   second, draining the output queue and updating all visual
   elements.
3. **Chunked processing** — Audio is segmented into 3-second chunks,
   balancing latency (≈ 3s feedback delay) against analysis quality.

This approach provides near-real-time feedback without requiring
WebSocket infrastructure or a custom frontend framework.

---

## 7  Intent Detection — Design Rationale

The intent detection system (Section 2.3) operates as a **zero-cost
intelligence layer** on top of existing keyword detection:

- It runs on the same lowercased text used for buying-signal
  detection — no additional preprocessing.
- It produces structured labels (`PRICING`, `OBJECTION`, etc.)
  that are immediately actionable for a salesperson.
- Multiple intents can fire simultaneously (e.g. a customer asking
  "how much does this cost compared to alternatives?" triggers both
  `PRICING` and `COMPARISON`).

### 7.1  Intent Confidence Scoring

Rather than treating intent detection as a binary signal, the system
computes a **continuous confidence score** using a length-damped
density formula:

```
density       = trigger_phrase_hits / word_count
length_factor = min(word_count / 10, 1.0)
confidence    = min(density × length_factor × 1.5, 1.0)
```

This addresses two edge cases:
1. **Very short utterances** ("how much?") — the `length_factor`
   dampens scores to prevent overconfidence on 1–2 word inputs.
2. **Long sentences with few matches** — density naturally decreases,
   reflecting lower intent concentration.

### 7.2  Temporal Smoothing

Raw per-segment confidence can jitter between chunks due to
natural variation in speech pacing. An **exponential moving average
(EMA)** smooths the signal:

```
smoothed = α × previous_confidence + (1 − α) × current_confidence
```

where `α = 0.7`. This produces stable, premium-feeling UI updates
without masking genuine shifts in conversational intent.

### 7.3  Intent → Recommendation Engine

Detected intents are mapped to **actionable coaching recommendations**,
transforming the system from a passive detection tool into an
**active decision support system**:

| Intent | Recommendation |
|--------|---------------|
| `PRICING` | Discuss pricing breakdown clearly — be transparent |
| `COMPARISON` | Highlight competitive differentiators proactively |
| `OBJECTION` | Address concerns directly — acknowledge and reframe |
| `COMMITMENT` | Reinforce decision — move to close or next steps |
| `INFORMATION` | Provide detailed walkthrough — offer a demo |

This establishes a three-stage intelligence pipeline:
**Detection → Interpretation → Action**.

### 7.4  Fusion Integration

Both the intent count and intent confidence are included in the
fusion vector, meaning the model can learn that higher intent
diversity *and* density correlate with higher engagement.  Intent
confidence directly influences the convincingness score through
both the trained Random Forest classifier and the heuristic
fallback (which boosts `COMMITMENT` and penalises `OBJECTION`).

---

## 8  Future Work

1. **Real-world dataset integration** — Train on RAVDESS and MELD
   to improve generalization beyond synthetic distributions.
2. **Cross-modal attention** — Replace feature concatenation with a
   Transformer-based fusion module that learns weighted cross-modal
   interactions.
3. **Deeper NLP** — Add named entity recognition, topic modelling,
   and discourse analysis for richer conversational understanding.
4. **Speaker diarisation** — Separate salesperson from customer to
   enable role-specific analysis.
5. **Adaptive thresholds** — Learn per-user engagement baselines
   to account for individual speaking styles.
6. **Edge deployment** — Optimise for on-device inference using
   ONNX Runtime or TensorFlow Lite for privacy-sensitive
   deployments.

---

## 9  Closing Statement

> *"Talklytics demonstrates how real-time multimodal AI systems
> can provide actionable feedback during conversations rather than
> post-analysis — shifting conversational intelligence from
> retrospective review to live decision support."*

The system's modular architecture, feature-level fusion approach,
and real-time simulation strategy establish a foundation that can
evolve from pipeline validation to production deployment as
real-world data and more sophisticated models are integrated.
