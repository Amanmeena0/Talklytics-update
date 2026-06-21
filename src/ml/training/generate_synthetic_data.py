"""
training/generate_synthetic_data.py
─────────────────────────────────────
Generates a synthetic labeled dataset for training the fusion model
when real audio datasets (RAVDESS / MELD) are not yet available.

Each sample is a feature vector (acoustic + linguistic) paired with a
Convincingness Score label (1–5).  Vectors are saved as a .npz file.

Usage
─────
python training/generate_synthetic_data.py --samples 1000 --output data/synthetic.npz
"""

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# Feature vector size (must match what the pipeline produces)
# AcousticFeatures.to_vector()  → mfcc_mean(13) + mfcc_std(13) + pitch_mean+std+energy(3) + spectral_contrast(7) = 36
# LinguisticFeatures.to_vector() → sentiment_encoded(1) + sentiment_score(1) + buying_count(1)
#                                   + hesitation_count(1) + intent_count(1) + intent_confidence(1) = 6
N_MFCC        = 13
ACOUSTIC_DIM  = N_MFCC * 2 + 3 + 7   # 36
LINGUISTIC_DIM = 6                    # matches LinguisticFeatures.to_vector()
TOTAL_DIM     = ACOUSTIC_DIM + LINGUISTIC_DIM   # 42


# ── Per-score statistical profiles ────────────────────────────────────── #
# Each profile defines (mean, std) for the key dimensions that
# discriminate between classes.

PROFILES = {
    #                energy   pitch  sentiment  buying  hesitation  intents  intent_conf
    1: dict(energy=0.01, pitch=80,  sentiment=-0.9, buying=0.0, hesitation=2.0, intents=0.0, intent_conf=0.0),
    2: dict(energy=0.02, pitch=100, sentiment=-0.3, buying=0.1, hesitation=1.0, intents=0.5, intent_conf=0.1),
    3: dict(energy=0.04, pitch=130, sentiment= 0.1, buying=0.2, hesitation=0.3, intents=1.0, intent_conf=0.3),
    4: dict(energy=0.07, pitch=160, sentiment= 0.6, buying=1.0, hesitation=0.1, intents=1.5, intent_conf=0.6),
    5: dict(energy=0.10, pitch=200, sentiment= 0.9, buying=2.0, hesitation=0.0, intents=2.0, intent_conf=0.9),
}


def _make_sample(score: int, rng: np.random.Generator) -> np.ndarray:
    p = PROFILES[score]

    vec = np.zeros(TOTAL_DIM, dtype=np.float32)

    # MFCC mean  (indices 0–12)
    vec[:N_MFCC] = rng.normal(0, 10, N_MFCC)

    # MFCC std  (indices 13–25)
    vec[N_MFCC:N_MFCC*2] = np.abs(rng.normal(3, 1, N_MFCC))

    # pitch_mean  (index 26)
    vec[N_MFCC*2]   = rng.normal(p["pitch"], 20)

    # pitch_std   (index 27)
    vec[N_MFCC*2+1] = abs(rng.normal(15, 5))

    # energy      (index 28)
    vec[N_MFCC*2+2] = abs(rng.normal(p["energy"], p["energy"] * 0.3))

    # spectral contrast (indices 29–35)
    vec[N_MFCC*2+3 : N_MFCC*2+10] = rng.normal(20, 5, 7)

    # linguistic features (indices 36–41) — must match LinguisticFeatures.to_vector() order:
    # [sentiment_encoded, sentiment_score, buying_count, hesitation_count, intent_count, intent_confidence]
    vec[ACOUSTIC_DIM]     = np.clip(rng.normal(p["sentiment"],    0.2), -1, 1)   # sentiment encoded (-1/0/1)
    vec[ACOUSTIC_DIM + 1] = abs(np.clip(rng.normal(0.7, 0.1), 0, 1))             # sentiment score (confidence)
    vec[ACOUSTIC_DIM + 2] = max(0, int(rng.normal(p["buying"],      0.5)))       # buying signal count
    vec[ACOUSTIC_DIM + 3] = max(0, int(rng.normal(p["hesitation"],  0.5)))       # hesitation count
    vec[ACOUSTIC_DIM + 4] = max(0, round(rng.normal(p["intents"],   0.4)))       # detected intent count
    vec[ACOUSTIC_DIM + 5] = float(np.clip(rng.normal(p["intent_conf"], 0.1), 0, 1))  # intent confidence

    return vec


def generate(n_samples: int, seed: int = 42) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    X, y = [], []

    per_class = n_samples // 5
    remainder = n_samples % 5

    for score in range(1, 6):
        count = per_class + (1 if score <= remainder else 0)
        for _ in range(count):
            X.append(_make_sample(score, rng))
            y.append(score)

    X = np.array(X, dtype=np.float32)
    y = np.array(y, dtype=np.int32)

    # Shuffle
    idx = rng.permutation(len(y))
    return X[idx], y[idx]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--samples", type=int, default=1000)
    parser.add_argument("--output",  type=str, default="data/synthetic.npz")
    parser.add_argument("--seed",    type=int, default=42)
    args = parser.parse_args()

    print(f"Generating {args.samples} synthetic samples …")
    X, y = generate(args.samples, seed=args.seed)

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    np.savez(args.output, X=X, y=y)
    print(f"Saved → {args.output}  (shape X={X.shape}, y={y.shape})")
    print(f"Class distribution: { {s: int((y==s).sum()) for s in range(1,6)} }")


if __name__ == "__main__":
    main()
