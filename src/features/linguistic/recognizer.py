"""
modules/speech_recognizer.py
─────────────────────────────
Converts a preprocessed audio segment to text using Faster-Whisper.
"""

import numpy as np
from faster_whisper import WhisperModel

from src.core.config import SAMPLE_RATE, WHISPER_MODEL_SIZE, WHISPER_LANGUAGE


class SpeechRecognizer:
    """Wraps Faster-Whisper for CPU-based real-time transcription."""

    def __init__(
        self,
        model_size: str = WHISPER_MODEL_SIZE,
        language:   str = WHISPER_LANGUAGE,
    ) -> None:
        self.language = language
        # compute_type="int8" keeps CPU usage reasonable for real-time use
        self._model = WhisperModel(model_size, device="cpu", compute_type="int8")

    def transcribe(self, segment: np.ndarray, sample_rate: int = SAMPLE_RATE) -> str:
        """Return the transcript for a single audio segment.

        Faster-Whisper accepts a float32 NumPy array directly.
        """
        segments, _ = self._model.transcribe(
            segment,
            language=self.language,
            beam_size=1,         # fastest mode for near-real-time
            vad_filter=True,     # skip silent parts automatically
        )
        text = " ".join(seg.text.strip() for seg in segments)
        return text.strip()
