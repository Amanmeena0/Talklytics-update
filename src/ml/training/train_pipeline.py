"""
training/train_model_standalone.py
────────────────────────────────────
Standalone training script that requires ONLY:
  numpy, scikit-learn, joblib

Does NOT import librosa / transformers / sounddevice.
Generates synthetic data → trains → evaluates → saves model.

Run with:
    python training/train_model_standalone.py
or:
    /path/to/venv/bin/python training/train_model_standalone.py
"""

import sys
import os
from pathlib import Path

import numpy as np
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# ── Project root on sys.path (so config.py can be read if needed) ─────── #
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# ── Output paths ──────────────────────────────────────────────────────── #
MODEL_PATH   = str(PROJECT_ROOT / "models" / "fusion_model.pkl")
ENCODER_PATH = str(PROJECT_ROOT / "models" / "label_encoder.pkl")

# ── Feature dimensionality (MUST match the live pipeline) ─────────────── #
# AcousticFeatures.to_vector():
#   mfcc_mean(13) + mfcc_std(13) + pitch_mean + pitch_std + energy(1) + spectral_contrast(7) = 36
# LinguisticFeatures.to_vector():
#   sentiment_encoded(1) + sentiment_score(1) + buying_count(1)
#   + hesitation_count(1) + intent_count(1) + intent_confidence(1) = 6
# Total = 42

N_MFCC         = 13
ACOUSTIC_DIM   = N_MFCC * 2 + 3 + 7   # 36
LINGUISTIC_DIM = 6
TOTAL_DIM      = ACOUSTIC_DIM + LINGUISTIC_DIM   # 42

# ── Per-class statistical profiles ────────────────────────────────────── #
PROFILES = {
    #           energy   pitch    sentiment  buying  hesitation  intents  intent_conf
    1: dict(energy=0.01, pitch=80,  sentiment=-0.9, buying=0.0, hesitation=2.0, intents=0.0, intent_conf=0.0),
    2: dict(energy=0.02, pitch=100, sentiment=-0.3, buying=0.1, hesitation=1.0, intents=0.5, intent_conf=0.1),
    3: dict(energy=0.04, pitch=130, sentiment= 0.1, buying=0.2, hesitation=0.3, intents=1.0, intent_conf=0.3),
    4: dict(energy=0.07, pitch=160, sentiment= 0.6, buying=1.0, hesitation=0.1, intents=1.5, intent_conf=0.6),
    5: dict(energy=0.10, pitch=200, sentiment= 0.9, buying=2.0, hesitation=0.0, intents=2.0, intent_conf=0.9),
}


def _make_sample(score: int, rng: np.random.Generator) -> np.ndarray:
    p   = PROFILES[score]
    vec = np.zeros(TOTAL_DIM, dtype=np.float32)

    # MFCC mean (indices 0–12)
    vec[:N_MFCC] = rng.normal(0, 10, N_MFCC)

    # MFCC std  (indices 13–25)
    vec[N_MFCC:N_MFCC * 2] = np.abs(rng.normal(3, 1, N_MFCC))

    # pitch_mean (index 26)
    vec[N_MFCC * 2]     = rng.normal(p["pitch"], 20)

    # pitch_std  (index 27)
    vec[N_MFCC * 2 + 1] = abs(rng.normal(15, 5))

    # energy     (index 28)
    vec[N_MFCC * 2 + 2] = abs(rng.normal(p["energy"], p["energy"] * 0.3 + 1e-6))

    # spectral contrast (indices 29–35)
    vec[N_MFCC * 2 + 3: N_MFCC * 2 + 10] = rng.normal(20, 5, 7)

    # ── Linguistic features (indices 36–41) ── #
    # Order matches LinguisticFeatures.to_vector() exactly
    vec[ACOUSTIC_DIM]     = float(np.clip(rng.normal(p["sentiment"],    0.2), -1, 1))   # sentiment_encoded
    vec[ACOUSTIC_DIM + 1] = float(abs(np.clip(rng.normal(0.7, 0.1), 0, 1)))             # sentiment_score
    vec[ACOUSTIC_DIM + 2] = float(max(0, int(rng.normal(p["buying"],      0.5))))       # buying_count
    vec[ACOUSTIC_DIM + 3] = float(max(0, int(rng.normal(p["hesitation"],  0.5))))       # hesitation_count
    vec[ACOUSTIC_DIM + 4] = float(max(0, round(rng.normal(p["intents"],   0.4))))       # intent_count
    vec[ACOUSTIC_DIM + 5] = float(np.clip(rng.normal(p["intent_conf"],    0.1), 0, 1))  # intent_confidence

    return vec


def generate(n_samples: int = 2_000, seed: int = 42):
    rng       = np.random.default_rng(seed)
    X, y      = [], []
    per_class = n_samples // 5
    remainder = n_samples % 5

    for score in range(1, 6):
        count = per_class + (1 if score <= remainder else 0)
        for _ in range(count):
            X.append(_make_sample(score, rng))
            y.append(score)

    X   = np.array(X, dtype=np.float32)
    y   = np.array(y, dtype=np.int32)
    idx = rng.permutation(len(y))
    return X[idx], y[idx]


def main() -> None:
    print("=" * 55)
    print("  ConvinceSense — Fusion Model Training")
    print("=" * 55)

    # 1. Generate
    print("\n[1/4] Generating synthetic dataset (2 000 samples) …")
    X, y = generate(n_samples=2_000, seed=42)
    print(f"      X: {X.shape}  y: {y.shape}  (feature_dim={TOTAL_DIM})")
    print(f"      Class distribution: { {s: int((y == s).sum()) for s in range(1, 6)} }")

    # 2. Split
    print("\n[2/4] Splitting 80/20 train/test …")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # 3. Train
    print("\n[3/4] Training Random Forest (200 trees, balanced weights) …")
    le  = LabelEncoder().fit(y_train)
    clf = RandomForestClassifier(
        n_estimators=200,
        random_state=42,
        class_weight="balanced",
        n_jobs=-1,
    )
    clf.fit(X_train, le.transform(y_train))

    # 4. Evaluate
    print("\n[4/4] Evaluating on held-out test set …")
    y_enc_pred = clf.predict(X_test)
    y_pred     = le.inverse_transform(y_enc_pred)

    acc = accuracy_score(y_test, y_pred)
    print(f"\n  Accuracy : {acc:.4f}")
    print("\n  Classification Report:")
    print(classification_report(y_test, y_pred, zero_division=0))
    print("  Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))

    # 5. Save
    os.makedirs(Path(MODEL_PATH).parent, exist_ok=True)
    joblib.dump(clf, MODEL_PATH)
    joblib.dump(le,  ENCODER_PATH)
    print(f"\n✅  Model saved  → {MODEL_PATH}")
    print(f"✅  Encoder saved → {ENCODER_PATH}")
    print("\n   Launch the dashboard:  streamlit run src/dashboard/app.py")


if __name__ == "__main__":
    main()
