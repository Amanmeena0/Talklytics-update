"""
modules/engagement_tracker.py
───────────────────────────────
Maintains the engagement timeline: a time-ordered list of
(timestamp, score, transcript, sentiment, buying_signals) records.
"""

import time
from dataclasses import dataclass, field

from src.core.config import INTENT_CONFIDENCE_SMOOTHING


@dataclass
class EngagementRecord:
    timestamp:        float
    score:            int
    transcript:       str
    sentiment:        str
    buying_signals:   list[str]
    hesitations:      list[str]
    detected_intents: list[str]
    intent_confidence: float
    recommendation:   str
    energy:           float
    confidence:       float


class EngagementTracker:
    """Append-only timeline of engagement records for one conversation."""

    def __init__(self) -> None:
        self._records: list[EngagementRecord] = []
        self._start: float = time.time()

    # ------------------------------------------------------------------ #
    #  Public API                                                          #
    # ------------------------------------------------------------------ #

    def add(
        self,
        score: int,
        transcript: str,
        sentiment: str,
        buying_signals: list[str],
        hesitations: list[str],
        detected_intents: list[str] | None = None,
        intent_confidence: float = 0.0,
        recommendation: str = "",
        energy: float = 0.0,
        confidence: float = 0.0,
    ) -> EngagementRecord:
        # ── Temporal smoothing (EMA) for intent confidence ─────────── #
        if self._records:
            prev = self._records[-1].intent_confidence
            alpha = INTENT_CONFIDENCE_SMOOTHING
            intent_confidence = alpha * prev + (1 - alpha) * intent_confidence

        rec = EngagementRecord(
            timestamp=time.time() - self._start,
            score=score,
            transcript=transcript,
            sentiment=sentiment,
            buying_signals=buying_signals,
            hesitations=hesitations,
            detected_intents=detected_intents or [],
            intent_confidence=round(intent_confidence, 3),
            recommendation=recommendation,
            energy=energy,
            confidence=confidence,
        )
        self._records.append(rec)
        return rec

    @property
    def records(self) -> list[EngagementRecord]:
        return list(self._records)

    @property
    def timestamps(self) -> list[float]:
        return [r.timestamp for r in self._records]

    @property
    def scores(self) -> list[int]:
        return [r.score for r in self._records]

    @property
    def average_score(self) -> float:
        if not self._records:
            return 0.0
        return sum(r.score for r in self._records) / len(self._records)

    @property
    def latest(self) -> EngagementRecord | None:
        return self._records[-1] if self._records else None

    def reset(self) -> None:
        self._records.clear()
        self._start = time.time()
