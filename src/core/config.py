# ─────────────────────────────────────────────
#  ConvinceSense — Central Configuration
# ─────────────────────────────────────────────

import os
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

# Load environment variables from .env if it exists
_env_path = _PROJECT_ROOT / ".env"
if _env_path.exists():
    with open(_env_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            # Skip empty lines or comments
            if not line or line.startswith("#"):
                continue
            if "=" in line:
                key, val = line.split("=", 1)
                key = key.strip()
                val = val.strip()
                # Strip quotes if present
                if (val.startswith('"') and val.endswith('"')) or (val.startswith("'") and val.endswith("'")):
                    val = val[1:-1]
                # Only set if not already set in actual environment
                if key not in os.environ:
                    os.environ[key] = val

# Audio
SAMPLE_RATE        = 16_000        # Hz
CHANNELS           = 1             # Mono
SEGMENT_DURATION   = 3             # seconds per analysis chunk
SILENCE_THRESHOLD  = 0.001         # RMS below this → segment skipped (lowered to catch quieter voices)

# Acoustic features
N_MFCC             = 13            # number of MFCC coefficients

# Whisper / ASR
WHISPER_MODEL_SIZE = "small"       # tiny | base | small | medium (upgraded to small for better accuracy)
WHISPER_LANGUAGE   = "en"

# NLP
SENTIMENT_MODEL    = "distilbert-base-uncased-finetuned-sst-2-english"

# Fusion model (platform-independent paths)
MODEL_PATH         = str(_PROJECT_ROOT / "src" / "ml" / "models" / "fusion_model.pkl")
LABEL_ENCODER_PATH = str(_PROJECT_ROOT / "src" / "ml" / "models" / "label_encoder.pkl")

# Score scale
SCORE_LABELS = {
    1: "Disengaged",
    2: "Low Interest",
    3: "Neutral",
    4: "Interested",
    5: "Highly Interested",
}

# Buying-signal keywords
BUYING_KEYWORDS = [
    "pricing", "price", "cost", "budget",
    "implementation", "integrate", "contract",
    "next steps", "timeline", "demo", "trial",
    "purchase", "buy", "sign", "proceed",
]

# Hesitation indicators
HESITATION_KEYWORDS = [
    "not sure", "maybe later", "expensive",
    "need to think", "not ready", "too much",
    "come back", "not right now", "busy",
]

# ── Intent detection patterns ──────────────────────────────────────────── #
# Rule-based intent classification for sales-conversation analysis.
# Each key is an intent label; each value is a list of trigger phrases.
INTENT_PATTERNS = {
    "PRICING": [
        "how much", "what's the cost", "price", "pricing",
        "budget", "afford", "rate", "fee", "charge",
        "subscription", "plan cost", "per month", "per year",
    ],
    "COMPARISON": [
        "compared to", "alternative", "better than", "versus",
        "competitor", "difference between", "how does it compare",
        "vs", "other options", "switch from",
    ],
    "OBJECTION": [
        "too expensive", "not convinced", "don't think",
        "not sure about", "concerned about", "worried",
        "risk", "don't need", "won't work", "not ready",
        "too complex", "difficult to",
    ],
    "COMMITMENT": [
        "let's do it", "sign me up", "ready to proceed",
        "let's go ahead", "i'm in", "move forward",
        "next steps", "get started", "set up a meeting",
        "send the contract", "let's close",
    ],
    "INFORMATION": [
        "tell me more", "how does it work", "explain",
        "can you show", "walk me through", "what exactly",
        "more details", "demo", "case study", "example",
    ],
}

# ── Intent → actionable recommendation ─────────────────────────────────── #
# Maps each detected intent to a salesperson-facing coaching recommendation.
INTENT_RECOMMENDATIONS = {
    "PRICING":     "💰 Discuss pricing breakdown clearly — be transparent.",
    "COMPARISON":  "⚖️ Highlight competitive differentiators proactively.",
    "OBJECTION":   "⚠️ Address concerns directly — acknowledge and reframe.",
    "COMMITMENT":  "✅ Reinforce decision — move to close or next steps.",
    "INFORMATION": "ℹ️ Provide detailed walkthrough — offer a demo.",
}

# ── Intent confidence tuning ───────────────────────────────────────────── #
INTENT_CONFIDENCE_SMOOTHING = 0.7   # EMA weight for temporal smoothing
