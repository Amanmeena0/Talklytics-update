"""
training/train_model.py
────────────────────────
CLI script to train and evaluate the ConvinceSense fusion model.

Usage examples
--------------
# Train on RAVDESS
python training/train_model.py --ravdess data/ravdess

# Train on a custom CSV
python training/train_model.py --csv data/labeled_calls.csv

# Combine both
python training/train_model.py --ravdess data/ravdess --csv data/labeled_calls.csv
"""

import argparse
import sys
from pathlib import Path

import numpy as np
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.ml.training.trainer import FusionTrainer
from src.ml.inference.fusion_inference  import FusionModel
from src.ml.training.data_loader  import DataLoader


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the ConvinceSense fusion model.")
    parser.add_argument("--ravdess", type=str, help="Path to RAVDESS root directory.")
    parser.add_argument("--csv",     type=str, help="Path to custom labeled CSV.")
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--seed",      type=int,   default=42)
    args = parser.parse_args()

    if not args.ravdess and not args.csv:
        parser.error("Provide at least one data source: --ravdess or --csv")

    loader = DataLoader()
    X_parts, y_parts = [], []

    if args.ravdess:
        print(f"Loading RAVDESS from {args.ravdess} …")
        Xr, yr = loader.load_ravdess(args.ravdess)
        X_parts.append(Xr); y_parts.append(yr)
        print(f"  → {len(yr)} samples loaded.")

    if args.csv:
        print(f"Loading CSV from {args.csv} …")
        Xc, yc = loader.load_csv(args.csv)
        X_parts.append(Xc); y_parts.append(yc)
        print(f"  → {len(yc)} samples loaded.")

    X = np.vstack(X_parts)
    y = np.concatenate(y_parts)
    print(f"\nTotal samples: {len(y)}")
    print(f"Feature vector size: {X.shape[1]}")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=args.test_size, random_state=args.seed, stratify=y
    )

    # Train
    trainer = FusionTrainer()
    print("\nTraining Random Forest …")
    trainer.train(X_train, y_train)

    # Save
    trainer.save()
    print("\nModel saved to models/")

    # Evaluate
    model = FusionModel()
    model.load()
    y_pred = np.array([model.predict_raw(x)[0] for x in X_test])

    print("\n── Evaluation Results ──────────────────────────────────────")
    print(f"Accuracy : {accuracy_score(y_test, y_pred):.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, zero_division=0))
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))


if __name__ == "__main__":
    main()
