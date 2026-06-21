"""
modules/fusion_model.py
────────────────────────
Random-Forest fusion model that combines acoustic + linguistic feature
vectors into a Convincingness Score (1–5).

Two modes
─────────
1. Inference only   – load a pre-trained model from disk.
2. Training         – call FusionModel.train() with labeled data then save.
"""

import os
from pathlib import Path

import numpy as np
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder

from src.core.config import MODEL_PATH, LABEL_ENCODER_PATH, SCORE_LABELS
from src.features.acoustic.extractor import AcousticFeatures
from src.features.linguistic.analyzer import LinguisticFeatures


class FusionModel:
    """Multimodal fusion model wrapping a scikit-learn RandomForest."""

    def __init__(self) -> None:
        self._clf: RandomForestClassifier | None = None
        self._le:  LabelEncoder | None = None

    # ------------------------------------------------------------------ #
    #  Public API                                                          #
    # ------------------------------------------------------------------ #

    def predict(
        self,
        acoustic: AcousticFeatures,
        linguistic: LinguisticFeatures,
    ) -> tuple[int, float]:
        """Return (score 1-5, confidence 0-1) for a single segment.

        Falls back to a heuristic if no trained model is available.
        """
        if self._clf is None:
            return self._heuristic(acoustic, linguistic)

        feature_vec = self._build_vector(acoustic, linguistic).reshape(1, -1)
        return self.predict_raw(feature_vec)

    def predict_raw(self, feature_vec: np.ndarray) -> tuple[int, float]:
        """Return (score 1-5, confidence 0-1) from a raw feature vector."""
        if self._clf is None:
            return 3, 0.0

        if len(feature_vec.shape) == 1:
            feature_vec = feature_vec.reshape(1, -1)

        label = self._clf.predict(feature_vec)[0]
        proba = max(self._clf.predict_proba(feature_vec)[0])
        score = int(self._le.inverse_transform([label])[0])
        return score, float(proba)

    def load(
        self,
        model_path: str  = MODEL_PATH,
        encoder_path: str = LABEL_ENCODER_PATH,
    ) -> None:
        """Load a previously saved model from disk."""
        if Path(model_path).exists() and Path(encoder_path).exists():
            self._clf = joblib.load(model_path)
            self._le  = joblib.load(encoder_path)



    # ------------------------------------------------------------------ #
    #  Internal helpers                                                    #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _build_vector(
        acoustic: AcousticFeatures,
        linguistic: LinguisticFeatures,
    ) -> np.ndarray:
        return np.concatenate([
            acoustic.to_vector(),
            linguistic.to_vector(),
        ])

    @staticmethod
    def _heuristic(
        acoustic: AcousticFeatures,
        linguistic: LinguisticFeatures,
    ) -> tuple[int, float]:
        """Simple rule-based fallback when no trained model is present."""
        score = 3  # neutral baseline

        if linguistic.sentiment_label == "POSITIVE":
            score += 1
        elif linguistic.sentiment_label == "NEGATIVE":
            score -= 1

        score += min(len(linguistic.buying_signals), 1)
        score -= min(len(linguistic.hesitations),    1)

        # Intent-aware scoring
        intents = set(linguistic.detected_intents)
        if "COMMITMENT" in intents:
            score += 1          # strong buying signal
        if "OBJECTION" in intents:
            score -= 1          # concern raised
        if len(intents) >= 2:
            score += 1          # high intent diversity = engaged customer

        if acoustic.energy > 0.05:
            score += 0  # could add energy-based rules here

        score = max(1, min(5, score))
        return score, 0.5
