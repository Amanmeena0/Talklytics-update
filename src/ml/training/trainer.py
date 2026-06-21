"""
src/ml/training/trainer.py
─────────────────────────
Handles the training logic for the ConvinceSense model.
"""

import os
from pathlib import Path

import numpy as np
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder

from src.core.config import MODEL_PATH, LABEL_ENCODER_PATH


class FusionTrainer:
    """Trains the Random-Forest fusion model."""

    def __init__(self) -> None:
        self.clf: RandomForestClassifier | None = None
        self.le: LabelEncoder | None = None

    def train(
        self,
        X: np.ndarray,
        y: np.ndarray,
        n_estimators: int = 200,
        random_state: int = 42,
    ) -> dict:
        """Train the fusion model.

        Parameters
        ----------
        X : shape (n_samples, n_features)  – concatenated feature vectors
        y : shape (n_samples,)             – integer labels 1–5
        """
        self.le = LabelEncoder().fit(y)
        y_enc = self.le.transform(y)
        self.clf = RandomForestClassifier(
            n_estimators=n_estimators,
            random_state=random_state,
            class_weight="balanced",
        )
        self.clf.fit(X, y_enc)
        return {"trained": True, "classes": list(self.le.classes_)}

    def save(
        self,
        model_path: str = MODEL_PATH,
        encoder_path: str = LABEL_ENCODER_PATH,
    ) -> None:
        """Persist model and label encoder to disk."""
        if self.clf is None or self.le is None:
            raise ValueError("Model is not trained yet.")
        os.makedirs(Path(model_path).parent, exist_ok=True)
        joblib.dump(self.clf, model_path)
        joblib.dump(self.le, encoder_path)
