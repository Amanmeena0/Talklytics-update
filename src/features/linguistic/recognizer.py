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
                "I'd be happy to explain. Talklytics is a real-time conversation intelligence platform that tracks buyer sentiment, intents, and signals.",
                "That sounds useful. How much does it cost? Is there a subscription fee or monthly plan?",
                "We offer custom packages starting at forty-nine dollars per seat, with flexible monthly or annual billing options.",
                "Compared to other options on the market, what makes Talklytics better?",
                "Unlike standard post-call analytics, we provide live recommendation cards and objection tracking during the call itself.",
                "I have a concern about the implementation timeline. Is it complex to integrate with our current CRM?",
                "Not at all. We support single-click integrations with major CRMs and VoIP dialers, taking less than an hour to set up.",
                "That sounds great, let's sign me up and get started.",
                "Fantastic! I will send over the contract and the onboarding next steps right away."
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
