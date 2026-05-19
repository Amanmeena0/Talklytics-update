# 🎙️ ConvinceSense
> Real-Time Conversational Interest Detection System

ConvinceSense analyses live sales calls and outputs a **Convincingness Score (1–5)**
that reflects how interested a customer sounds — using both *what* they say and *how* they say it.

---

## Architecture

```text
                     Microphone Input
                            ↓
                AudioCapture (sounddevice)
                            ↓
         AudioPreprocessor — normalise + silence filter
                            ↓
               ┌────────────┴────────────┐
               │                         │
       AcousticExtractor         SpeechRecognizer (Faster-Whisper)
       (librosa: MFCC,                   ↓
        pitch, energy)           LinguisticAnalyzer (DistilBERT)
               │                 (sentiment, keywords)
               └────────────┬────────────┘
                            ↓
               FusionModel (Random Forest)
                            ↓
             EngagementTracker → Streamlit Dashboard
```

---

## Quick Start

```bash
# 1. Create a virtual environment
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Launch the dashboard
streamlit run dashboard/app.py

# — or — run headless in the terminal
python main.py
```

The dashboard is served at http://localhost:8501

---

## Training the Fusion Model

```bash
# Using RAVDESS
python training/train_model.py --ravdess data/ravdess

# Using a custom CSV (columns: audio_path, transcript, score)
python training/train_model.py --csv data/labeled_calls.csv
```

Without a trained model the system uses a built-in heuristic fallback.

---

## Project Structure

```
convincesense/
├── config.py                   ← All tunable parameters
├── main.py                     ← Headless entry point
├── requirements.txt
├── modules/
│   ├── audio_capture.py        ← SoundDevice microphone streaming
│   ├── audio_preprocessor.py  ← Normalisation & silence filter
│   ├── acoustic_extractor.py  ← Librosa: MFCC, pitch, energy
│   ├── speech_recognizer.py   ← Faster-Whisper ASR
│   ├── linguistic_analyzer.py ← DistilBERT sentiment + keywords
│   ├── fusion_model.py        ← Random Forest classifier
│   ├── engagement_tracker.py  ← Timeline of engagement records
│   └── pipeline.py            ← Orchestrator (background thread)
├── dashboard/
│   └── app.py                 ← Streamlit real-time dashboard
├── training/
│   ├── data_loader.py         ← RAVDESS / CSV loaders
│   └── train_model.py         ← CLI training script
├── models/                    ← Saved model artifacts (.pkl)
└── data/                      ← Place datasets here
```

---

## Score Scale

| Score | Label             |
| ----- | ----------------- |
| 1     | Disengaged        |
| 2     | Low Interest      |
| 3     | Neutral           |
| 4     | Interested        |
| 5     | Highly Interested |

---

## Technology Stack

| Layer              | Technology               |
| ------------------ | ------------------------ |
| Language           | Python 3.9+              |
| Audio Capture      | sounddevice              |
| Audio Processing   | Librosa                  |
| Speech Recognition | Faster-Whisper           |
| NLP                | HuggingFace Transformers |
| Machine Learning   | Scikit-Learn             |
| Data Processing    | NumPy, Pandas            |
| Visualisation      | Matplotlib               |
| Dashboard          | Streamlit                |

---

## Hardware Requirements

| Component | Minimum               |
| --------- | --------------------- |
| CPU       | Intel i5 / AMD equiv. |
| RAM       | 8 GB                  |
| Storage   | 10 GB free            |
| GPU       | Optional              |

No GPU required — the system runs entirely on CPU.
