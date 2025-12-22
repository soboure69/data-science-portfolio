from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import streamlit as st
import tensorflow as tf


PROJECT_DIR = Path(__file__).resolve().parents[1]
MODEL_DIR = PROJECT_DIR / "models"
MODEL_PATH = MODEL_DIR / "sentiment_model.keras"
VECTORIZER_PATH = MODEL_DIR / "text_vectorizer.keras"


@st.cache_resource
def load_artifacts():
    model = tf.keras.models.load_model(MODEL_PATH)
    vectorizer = tf.keras.models.load_model(VECTORIZER_PATH)
    return model, vectorizer


def sigmoid_confidence(p: float) -> float:
    return float(max(p, 1.0 - p))


st.set_page_config(page_title="IMDB Sentiment", layout="centered")
st.title("IMDB Sentiment Analysis")
st.write("Predict if a movie review is positive or negative.")

if not MODEL_PATH.exists() or not VECTORIZER_PATH.exists():
    st.error(
        "Model artifacts not found. Train the model first and save to: "
        f"{MODEL_PATH} and {VECTORIZER_PATH}"
    )
    st.stop()

review = st.text_area("Review", height=200, placeholder="Type or paste a movie review...")

col1, col2 = st.columns(2)
with col1:
    run_btn = st.button("Predict", type="primary")
with col2:
    show_latency = st.checkbox("Show latency", value=True)

if run_btn:
    if not review.strip():
        st.warning("Please enter a review.")
        st.stop()

    model, vectorizer = load_artifacts()

    t0 = time.perf_counter()
    x = vectorizer(tf.constant([review]))
    p = float(model.predict(x, verbose=0)[0][0])
    dt_ms = (time.perf_counter() - t0) * 1000.0

    label = "Positive" if p >= 0.5 else "Negative"
    conf = sigmoid_confidence(p)

    st.subheader("Prediction")
    st.write(f"Label: **{label}**")
    st.write(f"Probability (positive): `{p:.3f}`")
    st.write(f"Confidence: `{conf:.3f}`")

    if show_latency:
        st.caption(f"Inference latency: {dt_ms:.1f} ms")
