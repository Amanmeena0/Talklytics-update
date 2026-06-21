"""
training/train_from_synthetic.py
──────────────────────────────────
Generate synthetic data → train → evaluate → save model.
This lets you validate the full training pipeline with zero external datasets.

Run with:
    python training/train_from_synthetic.py
"""

import sys
from pathlib import Path

import numpy as np
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.ml.training.generate_synthetic_data import generate
from src.ml.inference.fusion_inference import FusionModel


def main() -> None:
    print("=" * 55)
    print("  ConvinceSense — Synthetic Training Run")
    print("=" * 55)

    # 1. Generate
    print("\n[1/4] Generating synthetic dataset (2 000 samples) …")
    X, y = generate(n_samples=2_000, seed=42)
    print(f"      X: {X.shape}  y: {y.shape}")

    # 2. Split
    print("\n[2/4] Splitting 80/20 train/test …")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # 3. Train
    print("\n[3/4] Training Random Forest …")
    model = FusionModel()
    model.train(X_train, y_train)

    # 4. Evaluate
    print("\n[4/4] Evaluating …")

    # Use the internal RF directly for batch eval
    clf = model._clf
    le  = model._le
    y_enc_test = le.transform(y_test)
    y_enc_pred = clf.predict(X_test)
    y_pred = le.inverse_transform(y_enc_pred)

    acc = accuracy_score(y_test, y_pred)
    print(f"\n  Accuracy : {acc:.4f}")
    print("\n  Classification Report:")
    print(classification_report(y_test, y_pred, zero_division=0))
    print("  Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))

    # Save
    model.save()
    print("\n✅ Model saved to models/fusion_model.pkl")
    print("   Launch the dashboard: streamlit run src/dashboard/app.py")


if __name__ == "__main__":
    main()
