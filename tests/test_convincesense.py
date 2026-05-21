"""
tests/test_convincesense.py
────────────────────────────
Unit tests for all ConvinceSense modules.
Run with:  pytest tests/ -v
"""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import SAMPLE_RATE, N_MFCC
from modules.audio_preprocessor  import AudioPreprocessor
from modules.acoustic_extractor  import AcousticExtractor
from modules.linguistic_analyzer import LinguisticAnalyzer, LinguisticFeatures
from modules.fusion_model        import FusionModel
from modules.engagement_tracker  import EngagementTracker


# ── Fixtures ──────────────────────────────────────────────────────────── #

@pytest.fixture
def sine_segment():
    """A 3-second 440 Hz sine wave at 16 kHz."""
    t = np.linspace(0, 3, SAMPLE_RATE * 3, endpoint=False)
    return (0.3 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)


@pytest.fixture
def silence_segment():
    return np.zeros(SAMPLE_RATE * 3, dtype=np.float32)


# ── AudioPreprocessor ─────────────────────────────────────────────────── #

class TestAudioPreprocessor:
    def test_returns_array_for_valid_audio(self, sine_segment):
        pp = AudioPreprocessor()
        result = pp.process(sine_segment)
        assert result is not None
        assert isinstance(result, np.ndarray)

    def test_returns_none_for_silence(self, silence_segment):
        pp = AudioPreprocessor()
        result = pp.process(silence_segment)
        assert result is None

    def test_normalises_amplitude(self, sine_segment):
        pp = AudioPreprocessor()
        result = pp.process(sine_segment)
        assert np.max(np.abs(result)) <= 1.0 + 1e-6


# ── AcousticExtractor ─────────────────────────────────────────────────── #

class TestAcousticExtractor:
    def test_extract_returns_correct_mfcc_shape(self, sine_segment):
        extractor = AcousticExtractor()
        feats = extractor.extract(sine_segment)
        assert feats.mfcc_mean.shape == (N_MFCC,)
        assert feats.mfcc_std.shape  == (N_MFCC,)

    def test_energy_is_positive(self, sine_segment):
        extractor = AcousticExtractor()
        feats = extractor.extract(sine_segment)
        assert feats.energy > 0

    def test_to_vector_is_flat(self, sine_segment):
        extractor = AcousticExtractor()
        vec = extractor.extract(sine_segment).to_vector()
        assert vec.ndim == 1
        assert len(vec) == N_MFCC * 2 + 3 + 7   # 46


# ── LinguisticAnalyzer ────────────────────────────────────────────────── #

class TestLinguisticAnalyzer:
    @pytest.fixture(autouse=True)
    def analyzer(self):
        self.nlp = LinguisticAnalyzer()

    def test_positive_sentiment(self):
        feats = self.nlp.analyze("This looks great, I love the pricing!")
        assert feats.sentiment_label in ("POSITIVE", "NEUTRAL")

    def test_buying_signal_detection(self):
        feats = self.nlp.analyze("Can you walk me through the pricing and contract?")
        assert "pricing" in feats.buying_signals or "contract" in feats.buying_signals

    def test_hesitation_detection(self):
        feats = self.nlp.analyze("I'm not sure, it seems too expensive.")
        assert len(feats.hesitations) > 0

    def test_empty_string_returns_defaults(self):
        feats = self.nlp.analyze("")
        assert isinstance(feats, LinguisticFeatures)

    def test_to_vector_has_6_elements(self):
        feats = self.nlp.analyze("Let's discuss next steps.")
        vec = feats.to_vector()
        assert vec.shape == (6,)


# ── FusionModel (heuristic mode) ──────────────────────────────────────── #

class TestFusionModelHeuristic:
    def test_score_within_bounds(self, sine_segment):
        extractor = AcousticExtractor()
        nlp       = LinguisticAnalyzer()
        model     = FusionModel()   # no model loaded → heuristic

        acoustic   = extractor.extract(sine_segment)
        linguistic = nlp.analyze("That sounds interesting, what about pricing?")

        score, conf = model.predict(acoustic, linguistic)
        assert 1 <= score <= 5
        assert 0.0 <= conf <= 1.0

    def test_positive_text_scores_higher_than_negative(self, sine_segment):
        extractor = AcousticExtractor()
        nlp       = LinguisticAnalyzer()
        model     = FusionModel()

        acoustic = extractor.extract(sine_segment)
        pos_score, _ = model.predict(acoustic, nlp.analyze("Yes, I want to proceed and sign the contract!"))
        neg_score, _ = model.predict(acoustic, nlp.analyze("I'm not sure, maybe later, it's too expensive."))
        assert pos_score >= neg_score


# ── FusionModel (trained mode) ────────────────────────────────────────── #

class TestFusionModelTrained:
    def test_train_and_predict(self):
        from training.generate_synthetic_data import generate
        X, y = generate(n_samples=200, seed=0)
        model = FusionModel()
        model.train(X[:160], y[:160])
        score, conf = model._clf.predict(X[160:161]), None
        assert score is not None


# ── EngagementTracker ─────────────────────────────────────────────────── #

class TestEngagementTracker:
    def test_add_and_retrieve(self):
        tracker = EngagementTracker()
        rec = tracker.add(
            score=4,
            transcript="Sounds good.",
            sentiment="POSITIVE",
            buying_signals=["pricing"],
            hesitations=[],
            confidence=0.8,
        )
        assert tracker.latest.score == 4
        assert len(tracker.records) == 1

    def test_average_score(self):
        tracker = EngagementTracker()
        for s in [2, 4]:
            tracker.add(s, "", "NEUTRAL", [], [], 0.5)
        assert tracker.average_score == pytest.approx(3.0)

    def test_reset_clears_records(self):
        tracker = EngagementTracker()
        tracker.add(3, "", "NEUTRAL", [], [], 0.5)
        tracker.reset()
        assert len(tracker.records) == 0
        assert tracker.latest is None
