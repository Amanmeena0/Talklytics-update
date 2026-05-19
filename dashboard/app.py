"""
dashboard/app.py
─────────────────
ConvinceSense — Streamlit real-time dashboard.

Run with:
    streamlit run dashboard/app.py
"""

import queue
import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import streamlit as st

# Allow imports from project root
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import SCORE_LABELS
from modules.pipeline import ConvinceSensePipeline
from dashboard.sphere import render_reactive_sphere

# ── Page config ────────────────────────────────────────────────────────── #
st.set_page_config(
    page_title="ConvinceSense",
    page_icon="🎙️",
    layout="wide",
)

# ── Singleton pipeline (cached across reruns) ──────────────────────────── #
@st.cache_resource
def get_pipeline() -> ConvinceSensePipeline:
    return ConvinceSensePipeline()

pipeline = get_pipeline()

# ── Session state ──────────────────────────────────────────────────────── #
if "running"     not in st.session_state: st.session_state.running     = False
if "records"     not in st.session_state: st.session_state.records     = []
if "device_name" not in st.session_state: st.session_state.device_name = ""

# ── Sidebar status ─────────────────────────────────────────────────────── #
with st.sidebar:
    st.header("Status")
    if st.session_state.running:
        st.success("🎙️ Listening")
        if st.session_state.device_name:
            st.caption(f"Mic: {st.session_state.device_name}")
    else:
        st.info("⏸ Idle")

# ── Header ─────────────────────────────────────────────────────────────── #
st.title("🎙️ ConvinceSense")
st.caption("Real-Time Conversational Interest Detection")

# ── Controls ───────────────────────────────────────────────────────────── #
col_btn1, col_btn2, col_btn3 = st.columns([1, 1, 4])
with col_btn1:
    if st.button("▶ Start", disabled=st.session_state.running, use_container_width=True):
        pipeline.tracker.reset()
        st.session_state.records = []
        pipeline.start()
        st.session_state.running = True
        st.session_state.device_name = pipeline.capture.active_device_name
        st.rerun()

with col_btn2:
    if st.button("⏹ Stop", disabled=not st.session_state.running, use_container_width=True):
        pipeline.stop()
        st.session_state.running = False
        st.rerun()

# ── Reactive Listening Sphere ──────────────────────────────────────────── #
records = st.session_state.records

# Determine current energy level for the sphere
current_energy = 0.0
if records:
    current_energy = records[-1].energy or 0.0

render_reactive_sphere(
    energy=current_energy,
    is_listening=st.session_state.running,
)

st.divider()

# ── Drain the output queue ─────────────────────────────────────────────── #
if st.session_state.running:
    drained = 0
    while drained < 10:
        try:
            record = pipeline.output_q.get_nowait()
            st.session_state.records.append(record)
            drained += 1
        except queue.Empty:
            break
    # Update records reference after drain
    records = st.session_state.records

# ── Layout: metrics + latest segment ──────────────────────────────────── #
col_score, col_avg, col_sentiment, col_keywords = st.columns(4)

if records:
    latest = records[-1]
    avg    = pipeline.tracker.average_score

    col_score.metric(
        "Current Score",
        f"{latest.score} / 5",
        delta=SCORE_LABELS.get(latest.score, ""),
    )
    col_avg.metric("Session Average", f"{avg:.1f} / 5")
    col_sentiment.metric("Sentiment", latest.sentiment)
    col_keywords.metric(
        "Buying Signals",
        len(latest.buying_signals),
        delta=f"+{len(latest.buying_signals)}" if latest.buying_signals else None,
    )

    # Latest transcript
    st.subheader("Latest Transcript")
    st.info(latest.transcript or "_(no speech detected)_")

    if latest.buying_signals:
        st.success("🟢 Buying signals: " + ", ".join(latest.buying_signals))
    if latest.hesitations:
        st.warning("🟡 Hesitation indicators: " + ", ".join(latest.hesitations))

    # ── Intent badges + recommendation ──────────────────────────────── #
    intents = latest.detected_intents or []
    conf    = latest.intent_confidence or 0.0

    if intents:
        _INTENT_EMOJI = {
            "PRICING":     "💰",
            "COMPARISON":  "⚖️",
            "OBJECTION":   "⚠️",
            "COMMITMENT":  "✅",
            "INFORMATION": "ℹ️",
        }
        badges = "  ".join(
            f"{_INTENT_EMOJI.get(i, '🔹')} **{i.title()}**"
            for i in intents
        )
        conf_pct = f"{conf * 100:.0f}%"
        st.markdown(
            f"🎯 **Detected Intents** (confidence: {conf_pct}):  {badges}"
        )

    if latest.recommendation:
        st.info(f"💡 **Recommendation:** {latest.recommendation}")

else:
    col_score.metric("Current Score", "—")
    col_avg.metric("Session Average", "—")
    col_sentiment.metric("Sentiment", "—")
    col_keywords.metric("Buying Signals", "—")
    st.info("Press **▶ Start** and speak into your microphone.")

st.divider()

# ── Engagement timeline chart ─────────────────────────────────────────── #
st.subheader("📈 Engagement Timeline")

if len(records) >= 2:
    timestamps = [r.timestamp for r in records]
    scores     = [r.score     for r in records]

    fig, ax = plt.subplots(figsize=(10, 3))
    ax.plot(timestamps, scores, marker="o", color="#4F8EF7", linewidth=2)
    ax.fill_between(timestamps, scores, alpha=0.15, color="#4F8EF7")
    ax.set_ylim(0.5, 5.5)
    ax.set_yticks(range(1, 6))
    ax.set_yticklabels([f"{i} – {SCORE_LABELS[i]}" for i in range(1, 6)], fontsize=8)
    ax.set_xlabel("Time (s)")
    ax.set_title("Convincingness Score over Conversation")
    ax.grid(axis="y", linestyle="--", alpha=0.5)
    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)
else:
    st.caption("Chart will appear after 2 or more segments are analysed.")

# ── Transcript log ─────────────────────────────────────────────────────── #
if records:
    with st.expander("📝 Full Transcript Log", expanded=False):
        for rec in reversed(records):
            mins, secs = divmod(int(rec.timestamp), 60)
            intent_str = ""
            if rec.detected_intents:
                intent_str = " | 🎯 " + ", ".join(rec.detected_intents)
            st.markdown(
                f"**{mins:02d}:{secs:02d}** | Score `{rec.score}` | "
                f"{rec.sentiment}{intent_str} | _{rec.transcript}_"
            )

# ── Auto-refresh while running ─────────────────────────────────────────── #
if st.session_state.running:
    time.sleep(1)
    st.rerun()
