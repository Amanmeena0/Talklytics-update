"""
modules/speech_recognizer.py
─────────────────────────────
Converts a preprocessed audio segment to text using Faster-Whisper.
"""

import os
import numpy as np

from src.core.config import SAMPLE_RATE, WHISPER_MODEL_SIZE, WHISPER_LANGUAGE


class SpeechRecognizer:
    """Wraps Faster-Whisper for CPU-based real-time transcription."""

    def __init__(
        self,
        model_size: str = WHISPER_MODEL_SIZE,
        language:   str = WHISPER_LANGUAGE,
    ) -> None:
        self.model_size = model_size
        self.language = language
        self._model = None

    def transcribe(self, segment: np.ndarray, sample_rate: int = SAMPLE_RATE) -> str:
        """Return the transcript for a single audio segment.

        Faster-Whisper accepts a float32 NumPy array directly.
        """
        if self._model is None:
            if os.getenv("RENDER") == "true" or os.getenv("LIGHTWEIGHT_MODE") == "true":
                print("[SpeechRecognizer] Running in lightweight mode. Using simulated transcription.")
                self._model = "simulated"
            else:
                from faster_whisper import WhisperModel
                self._model = WhisperModel(self.model_size, device="cpu", compute_type="int8")

        if self._model == "simulated":
            if not hasattr(self, "_step"):
                self._step = 0
            
            script = [
                "Hello, I am interested in hearing more about your Talklytics platform.",
                "How much does it cost? Is there a subscription fee or monthly plan?",
                "Compared to other options, what makes it better?",
                "That sounds great, let's sign me up and get started.",
                "I have a concern about the implementation timeline, is it too complex?"
            ]
            
            rms = np.sqrt(np.mean(segment**2))
            if rms < 0.001:
                return ""
            
            text = script[self._step % len(script)]
            self._step += 1
            return text

        segments, _ = self._model.transcribe(
            segment,
            language=self.language,
            beam_size=1,         # fastest mode for near-real-time
            vad_filter=True,     # skip silent parts automatically
            condition_on_previous_text=False,
        )
        text = " ".join(seg.text.strip() for seg in segments)
        return text.strip()
