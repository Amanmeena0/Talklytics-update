"""
modules/acoustic_extractor.py
──────────────────────────────
Extracts acoustic / prosodic features from a preprocessed audio segment
using Librosa: MFCCs, pitch, energy, and spectral contrast.
"""

from dataclasses import dataclass, field

import numpy as np
import librosa

from config import SAMPLE_RATE, N_MFCC


@dataclass
class AcousticFeatures:
    mfcc_mean: np.ndarray = field(default_factory=lambda: np.zeros(N_MFCC))
    mfcc_std:  np.ndarray = field(default_factory=lambda: np.zeros(N_MFCC))
    pitch_mean: float = 0.0
    pitch_std:  float = 0.0
    energy:     float = 0.0
    spectral_contrast_mean: np.ndarray = field(default_factory=lambda: np.zeros(7))

    def to_vector(self) -> np.ndarray:
        """Flatten all features into a single 1-D vector."""
        return np.concatenate([
            self.mfcc_mean,
            self.mfcc_std,
            [self.pitch_mean, self.pitch_std, self.energy],
            self.spectral_contrast_mean,
        ])


class AcousticExtractor:
    """Extracts prosodic features from a float32 audio array."""

    def __init__(self, sample_rate: int = SAMPLE_RATE, n_mfcc: int = N_MFCC) -> None:
        self.sample_rate = sample_rate
        self.n_mfcc      = n_mfcc

    def extract(self, segment: np.ndarray) -> AcousticFeatures:
        y = segment.astype(np.float32)
        sr = self.sample_rate

        # ── MFCCs ────────────────────────────────────────────────────── #
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=self.n_mfcc)

        # ── Pitch (fundamental frequency via pyin) ────────────────────── #
        f0, voiced_flag, _ = librosa.pyin(
            y,
            fmin=librosa.note_to_hz("C2"),
            fmax=librosa.note_to_hz("C7"),
            sr=sr,
        )
        f0_voiced = f0[voiced_flag] if voiced_flag is not None and voiced_flag.any() else np.array([0.0])

        # ── RMS Energy ───────────────────────────────────────────────── #
        rms = librosa.feature.rms(y=y)

        # ── Spectral Contrast ─────────────────────────────────────────── #
        spectral_contrast = librosa.feature.spectral_contrast(y=y, sr=sr)

        return AcousticFeatures(
            mfcc_mean=mfcc.mean(axis=1),
            mfcc_std=mfcc.std(axis=1),
            pitch_mean=float(np.mean(f0_voiced)),
            pitch_std=float(np.std(f0_voiced)),
            energy=float(rms.mean()),
            spectral_contrast_mean=spectral_contrast.mean(axis=1),
        )
