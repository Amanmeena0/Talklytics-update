"""
modules/audio_preprocessor.py
──────────────────────────────
Normalises amplitude and filters out silent segments.
"""

import numpy as np
from src.core.config import SILENCE_THRESHOLD


class AudioPreprocessor:
    """Prepares a raw audio segment for feature extraction."""

    def __init__(self, silence_threshold: float = SILENCE_THRESHOLD) -> None:
        self.silence_threshold = silence_threshold

    def process(self, segment: np.ndarray) -> np.ndarray | None:
        """Normalise and silence-check a segment.

        Returns
        -------
        np.ndarray  – processed segment
        None        – segment is silent and should be skipped
        """
        segment = segment.astype(np.float32)

        # Silence filter
        rms = float(np.sqrt(np.mean(segment ** 2)))
        if rms < self.silence_threshold:
            return None

        # Amplitude normalisation
        peak = np.max(np.abs(segment))
        if peak > 0:
            segment = segment / peak

        return segment
