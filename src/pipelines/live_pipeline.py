"""
modules/pipeline.py
────────────────────
ConvinceSense analysis pipeline.
Ties together audio capture → preprocessing → feature extraction →
ASR → NLP → fusion model → engagement tracker.

The pipeline runs in a background thread and pushes EngagementRecord
objects into an output queue that the Streamlit dashboard consumes.
"""

import queue
import threading

import numpy as np

from src.core.config import INTENT_RECOMMENDATIONS
from src.core.audio_capture       import AudioCapture
from src.features.acoustic.preprocessor  import AudioPreprocessor
from src.features.acoustic.extractor  import AcousticExtractor
from src.features.linguistic.recognizer   import SpeechRecognizer
from src.features.linguistic.analyzer import LinguisticAnalyzer
from src.ml.inference.fusion_inference        import FusionModel
from src.features.engagement.tracker  import EngagementTracker, EngagementRecord


class ConvinceSensePipeline:
    """Full real-time analysis pipeline."""

    def __init__(self) -> None:
        # Modules
        self.capture     = AudioCapture()
        self.preprocessor= AudioPreprocessor()
        self.acoustic    = AcousticExtractor()
        self.asr         = SpeechRecognizer()
        self.nlp         = LinguisticAnalyzer()
        self.model       = FusionModel()
        self.tracker     = EngagementTracker()

        # Output queue consumed by the dashboard
        self.output_q: queue.Queue[EngagementRecord] = queue.Queue()

        self._thread: threading.Thread | None = None
        self._running = False

        # Try to load a pre-trained model
        self.model.load()

    # ------------------------------------------------------------------ #
    #  Public API                                                          #
    # ------------------------------------------------------------------ #

    def start(self) -> None:
        """Start the audio capture and processing thread."""
        self._running = True
        self.capture.start()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        """Stop all processing."""
        self._running = False
        self.capture.stop()
        if self._thread:
            self._thread.join(timeout=5)

    # ------------------------------------------------------------------ #
    #  Internal processing loop                                           #
    # ------------------------------------------------------------------ #

    def _run(self) -> None:
        print("[Pipeline] Processing thread started.")
        try:
            while self._running:
                # 1. Capture
                segment = self.capture.get_segment(timeout=5.0)
                if segment is None:
                    continue

                try:
                    # 2. Preprocess
                    clean = self.preprocessor.process(segment)
                    if clean is None:
                        continue  # silent segment — skip quietly

                    # 3. Acoustic features
                    acoustic_features = self.acoustic.extract(clean)

                    # 4. ASR transcription
                    transcript = self.asr.transcribe(clean)

                    # 5. Linguistic analysis
                    linguistic_features = self.nlp.analyze(transcript)

                    # 6. Fusion → score
                    score, confidence = self.model.predict(acoustic_features, linguistic_features)

                    # 7. Generate actionable recommendation
                    intents = linguistic_features.detected_intents
                    recommendation = ""
                    if intents:
                        for intent in intents:
                            if intent in INTENT_RECOMMENDATIONS:
                                recommendation = INTENT_RECOMMENDATIONS[intent]
                                break

                    # 8. Track and emit
                    record = self.tracker.add(
                        score=score,
                        transcript=transcript,
                        sentiment=linguistic_features.sentiment_label,
                        buying_signals=linguistic_features.buying_signals,
                        hesitations=linguistic_features.hesitations,
                        detected_intents=intents,
                        intent_confidence=linguistic_features.intent_confidence,
                        recommendation=recommendation,
                        energy=acoustic_features.energy,
                        confidence=confidence,
                    )
                    self.output_q.put(record)
                    print(f"[Pipeline] ✅ Score={score} | '{transcript[:60]}'")

                except Exception as e:
                    # Log the error but keep the thread alive for the next segment
                    print(f"[Pipeline] ⚠️  Segment error (skipping): {type(e).__name__}: {e}")

        except Exception as fatal:
            print(f"[Pipeline] 💥 Fatal thread error: {type(fatal).__name__}: {fatal}")

