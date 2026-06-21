"""
modules/linguistic_analyzer.py
────────────────────────────────
Analyses a text transcript for sentiment and buying/hesitation signals
using HuggingFace DistilBERT.
"""

from dataclasses import dataclass, field

import numpy as np
from transformers import pipeline

from src.core.config import (
    SENTIMENT_MODEL,
    BUYING_KEYWORDS,
    HESITATION_KEYWORDS,
    INTENT_PATTERNS,
)


@dataclass
class LinguisticFeatures:
    sentiment_label:  str = "NEUTRAL"   # POSITIVE | NEGATIVE | NEUTRAL
    sentiment_score:  float = 0.0       # confidence in [0, 1]
    buying_signals:   list[str] = field(default_factory=list)
    hesitations:      list[str] = field(default_factory=list)
    detected_intents: list[str] = field(default_factory=list)
    intent_confidence: float = 0.0      # ratio of trigger-phrase hits to word count

    def to_vector(self) -> np.ndarray:
        """Encode as a numeric vector for the fusion model.

        [sentiment_encoded, sentiment_score, buying_count, hesitation_count,
         intent_count, intent_confidence]
        """
        label_map = {"POSITIVE": 1, "NEUTRAL": 0, "NEGATIVE": -1}
        return np.array([
            label_map.get(self.sentiment_label, 0),
            self.sentiment_score,
            len(self.buying_signals),
            len(self.hesitations),
            len(self.detected_intents),
            self.intent_confidence,
        ], dtype=np.float32)


class LinguisticAnalyzer:
    """Runs sentiment analysis and keyword detection on a transcript."""

    def __init__(self, model_name: str = SENTIMENT_MODEL) -> None:
        self._sentiment = pipeline(
            "sentiment-analysis",
            model=model_name,
            truncation=True,
            max_length=512,
        )

    def analyze(self, text: str) -> LinguisticFeatures:
        if not text.strip():
            return LinguisticFeatures()

        # ── Sentiment ─────────────────────────────────────────────────── #
        result = self._sentiment(text)[0]
        label  = result["label"].upper()   # POSITIVE / NEGATIVE
        score  = float(result["score"])

        # Normalise to three-class
        if label == "NEGATIVE" and score < 0.65:
            label = "NEUTRAL"

        # ── Keyword detection ─────────────────────────────────────────── #
        lower = text.lower()
        buying    = [kw for kw in BUYING_KEYWORDS    if kw in lower]
        hesitation= [kw for kw in HESITATION_KEYWORDS if kw in lower]

        # ── Intent detection ──────────────────────────────────────────── #
        intents, intent_conf = self._detect_intents(lower)

        return LinguisticFeatures(
            sentiment_label=label,
            sentiment_score=score,
            buying_signals=buying,
            hesitations=hesitation,
            detected_intents=intents,
            intent_confidence=intent_conf,
        )

    # ------------------------------------------------------------------ #
    #  Intent detection (rule-based)                                       #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _detect_intents(text_lower: str) -> tuple[list[str], float]:
        """Return matched intent labels and a confidence score.

        Confidence uses density × length-dampening so that:
        - Very short utterances don't score artificially high.
        - Long sentences aren't unfairly diluted.
        """
        matched: list[str] = []
        total_hits = 0
        for intent_label, phrases in INTENT_PATTERNS.items():
            hits = sum(1 for phrase in phrases if phrase in text_lower)
            if hits:
                matched.append(intent_label)
                total_hits += hits

        word_count = max(len(text_lower.split()), 1)
        if total_hits == 0:
            return matched, 0.0

        density       = total_hits / word_count
        length_factor = min(word_count / 10, 1.0)   # damp ≤10-word inputs
        confidence    = min(density * length_factor * 1.5, 1.0)
        return matched, round(confidence, 3)
