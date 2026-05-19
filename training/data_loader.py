"""
training/data_loader.py
────────────────────────
Helpers for loading and labelling training data from:
  • RAVDESS emotional speech dataset
  • MELD (multimodal sentiment) dataset
  • Custom CSV with (audio_path, label) columns

All loaders return (X, y) where
  X : np.ndarray shape (n_samples, n_features)
  y : np.ndarray shape (n_samples,)  – integer labels 1–5
"""

import os
from pathlib import Path

import numpy as np
import pandas as pd
import librosa

from modules.acoustic_extractor  import AcousticExtractor
from modules.linguistic_analyzer import LinguisticAnalyzer


# ── RAVDESS emotion → ConvinceSense score mapping ──────────────────────── #
# RAVDESS emotion codes: 01=neutral 02=calm 03=happy 04=sad
#                        05=angry  06=fearful 07=disgust 08=surprised
RAVDESS_TO_SCORE = {
    "01": 3,  # neutral
    "02": 3,  # calm
    "03": 5,  # happy  → Highly Interested
    "04": 1,  # sad    → Disengaged
    "05": 2,  # angry  → Low Interest
    "06": 2,  # fearful
    "07": 1,  # disgust
    "08": 4,  # surprised → Interested
}


class DataLoader:
    def __init__(self, sample_rate: int = 16_000) -> None:
        self.sample_rate = sample_rate
        self._acoustic   = AcousticExtractor(sample_rate=sample_rate)
        self._linguistic = LinguisticAnalyzer()

    # ------------------------------------------------------------------ #
    #  RAVDESS                                                             #
    # ------------------------------------------------------------------ #

    def load_ravdess(self, directory: str) -> tuple[np.ndarray, np.ndarray]:
        """Load .wav files from a RAVDESS directory."""
        X, y = [], []
        for path in sorted(Path(directory).rglob("*.wav")):
            parts  = path.stem.split("-")
            if len(parts) < 3:
                continue
            emotion_code = parts[2]
            score = RAVDESS_TO_SCORE.get(emotion_code, 3)

            audio, _ = librosa.load(str(path), sr=self.sample_rate, mono=True)
            feats = self._acoustic.extract(audio)

            # No text available → use zero linguistic vector
            from modules.linguistic_analyzer import LinguisticFeatures
            ling = LinguisticFeatures()

            vec = np.concatenate([feats.to_vector(), ling.to_vector()])
            X.append(vec)
            y.append(score)

        return np.array(X), np.array(y)

    # ------------------------------------------------------------------ #
    #  Custom CSV                                                          #
    # ------------------------------------------------------------------ #

    def load_csv(self, csv_path: str, audio_col: str = "audio_path", label_col: str = "score") -> tuple[np.ndarray, np.ndarray]:
        """Load from a CSV with columns: audio_path, transcript (opt), score."""
        df = pd.read_csv(csv_path)
        X, y = [], []

        for _, row in df.iterrows():
            audio, _ = librosa.load(row[audio_col], sr=self.sample_rate, mono=True)
            acoustic  = self._acoustic.extract(audio)

            transcript = row.get("transcript", "")
            linguistic = self._linguistic.analyze(str(transcript)) if transcript else None

            from modules.linguistic_analyzer import LinguisticFeatures
            ling_vec = (linguistic or LinguisticFeatures()).to_vector()

            vec = np.concatenate([acoustic.to_vector(), ling_vec])
            X.append(vec)
            y.append(int(row[label_col]))

        return np.array(X), np.array(y)
