"""
main.py
────────
ConvinceSense headless entry point.
Runs the full pipeline and prints engagement records to the terminal.
Use this for testing without Streamlit.

Run with:
    python main.py
"""

import signal
import sys
import time

from src.core.config import SCORE_LABELS
from src.pipelines.live_pipeline import ConvinceSensePipeline


def main() -> None:
    pipeline = ConvinceSensePipeline()

    print("ConvinceSense — Real-Time Conversational Interest Detection")
    print("Press Ctrl+C to stop.\n")

    pipeline.start()

    def _shutdown(sig, frame):
        print("\nStopping …")
        pipeline.stop()
        sys.exit(0)

    signal.signal(signal.SIGINT, _shutdown)

    while True:
        record = pipeline.output_q.get(timeout=30)
        label  = SCORE_LABELS.get(record.score, "Unknown")
        print(
            f"[{record.timestamp:6.1f}s] "
            f"Score: {record.score}/5 ({label:18s}) | "
            f"Sentiment: {record.sentiment:8s} | "
            f"Transcript: {record.transcript[:60]}"
        )
        if record.buying_signals:
            print(f"          🟢 Buying signals : {record.buying_signals}")
        if record.hesitations:
            print(f"          🟡 Hesitations    : {record.hesitations}")
        if record.detected_intents:
            intents = record.detected_intents or []
            conf = f"{(record.intent_confidence or 0.0) * 100:.0f}%"
            print(f"          🎯 Intents ({conf:>4s}) : {intents}")
        if record.recommendation:
            print(f"          💡 Recommend       : {record.recommendation}")


if __name__ == "__main__":
    main()
