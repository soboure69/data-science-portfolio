from __future__ import annotations

import time
from pathlib import Path
import os
import sys
import urllib.request
import zipfile

import numpy as np
import streamlit as st
import tensorflow as tf


PROJECT_DIR = Path(__file__).resolve().parents[1]
MODEL_DIR = PROJECT_DIR / "models"
MODEL_PATH = MODEL_DIR / "sentiment_model.keras"
VECTORIZER_PATH = MODEL_DIR / "text_vectorizer.keras"

sys.path.insert(0, str(PROJECT_DIR))

from src.text_preprocessing import TextPreprocessor


def download_file(url: str, dest_path: Path) -> None:
    dest_path.parent.mkdir(parents=True, exist_ok=True)

    def reporthook(block_num: int, block_size: int, total_size: int):
        if total_size <= 0:
            return
        downloaded = block_num * block_size
        progress = min(downloaded / total_size, 1.0)
        progress_bar.progress(progress)

    progress_bar = st.progress(0.0)
    urllib.request.urlretrieve(url, dest_path, reporthook=reporthook)
    progress_bar.empty()


def is_valid_keras_artifact(path: Path) -> bool:
    if not path.exists():
        return False
    if path.suffix != ".keras":
        return True
    return zipfile.is_zipfile(path)


def ensure_artifact(path: Path, url: str | None) -> None:
    if is_valid_keras_artifact(path):
        return
    if path.exists():
        try:
            path.unlink()
        except OSError:
            pass
    if not url:
        raise FileNotFoundError(f"Missing artifact: {path}")
    download_file(url, path)

    if not is_valid_keras_artifact(path):
        raise ValueError(
            "Downloaded artifact is not a valid .keras file (expected a Keras zip archive). "
            "Double-check that your URL is a direct GitHub Release asset URL."
        )


@st.cache_resource
def load_artifacts(model_path: str, vectorizer_path: str):
    model = tf.keras.models.load_model(model_path, compile=False)
    vectorizer = tf.keras.models.load_model(
        vectorizer_path,
        compile=False,
        custom_objects={"_custom_standardize": TextPreprocessor._custom_standardize},
    )
    return model, vectorizer


def sigmoid_confidence(p: float) -> float:
    return float(max(p, 1.0 - p))


st.set_page_config(page_title="IMDB Sentiment", layout="centered")
st.title("IMDB Sentiment Analysis")
st.write("Predict if a movie review is positive or negative.")

threshold = st.sidebar.slider("Decision threshold (positive)", 0.05, 0.95, 0.50, 0.05)

DEFAULT_MODEL_URL = (
    "https://github.com/soboure69/data-science-portfolio/releases/download/v1.0/"
    "sentiment_model.keras"
)
DEFAULT_VECTORIZER_URL = (
    "https://github.com/soboure69/data-science-portfolio/releases/download/v1.0/"
    "text_vectorizer.keras"
)

secrets_model_url = None
secrets_vectorizer_url = None
try:
    secrets_model_url = st.secrets.get("MODEL_URL")
    secrets_vectorizer_url = st.secrets.get("VECTORIZER_URL")
except Exception:
    secrets_model_url = None
    secrets_vectorizer_url = None

env_model_url = os.environ.get("MODEL_URL", "")
env_vectorizer_url = os.environ.get("VECTORIZER_URL", "")

default_model_url = secrets_model_url or env_model_url or DEFAULT_MODEL_URL
default_vectorizer_url = secrets_vectorizer_url or env_vectorizer_url or DEFAULT_VECTORIZER_URL

local_artifacts_present = MODEL_PATH.exists() and VECTORIZER_PATH.exists()
urls_present = bool(secrets_model_url or secrets_vectorizer_url or env_model_url or env_vectorizer_url)

default_mode = "Local" if local_artifacts_present and not urls_present else "GitHub Release"
mode = st.sidebar.selectbox(
    "Artifacts mode",
    ["Local", "GitHub Release"],
    index=0 if default_mode == "Local" else 1,
)

model_url = None
vectorizer_url = None

if mode == "GitHub Release":
    st.sidebar.write("Provide direct download URLs (e.g. GitHub Release asset URLs).")
    model_url = st.sidebar.text_input("MODEL_URL", value=default_model_url)
    vectorizer_url = st.sidebar.text_input("VECTORIZER_URL", value=default_vectorizer_url)

try:
    ensure_artifact(MODEL_PATH, model_url)
    ensure_artifact(VECTORIZER_PATH, vectorizer_url)
except FileNotFoundError as e:
    st.error(
        "Model artifacts not found.\n\n"
        f"{e}\n\n"
        "If you're using GitHub Release mode, set MODEL_URL and VECTORIZER_URL."
    )
    st.stop()

examples = {
    "Positive example": "This movie was surprisingly good, I loved the acting and the story.",
    "Negative example": "Terrible movie. The plot made no sense and the acting was awful.",
}

if "review" not in st.session_state:
    st.session_state["review"] = ""

ex_col1, ex_col2, ex_col3 = st.columns([1, 1, 1])
with ex_col1:
    if st.button("Use positive example"):
        st.session_state["review"] = examples["Positive example"]
with ex_col2:
    if st.button("Use negative example"):
        st.session_state["review"] = examples["Negative example"]
with ex_col3:
    if st.button("Clear"):
        st.session_state["review"] = ""

review = st.text_area(
    "Review",
    key="review",
    height=200,
    placeholder="Type or paste a movie review...",
)

col1, col2 = st.columns(2)
with col1:
    run_btn = st.button("Predict", type="primary")
with col2:
    show_latency = st.checkbox("Show latency", value=True)

if run_btn:
    if not review.strip():
        st.warning("Please enter a review.")
        st.stop()

    model, vectorizer = load_artifacts(str(MODEL_PATH), str(VECTORIZER_PATH))

    t0 = time.perf_counter()
    x = vectorizer(tf.constant([review]))
    p = float(model.predict(x, verbose=0)[0][0])
    dt_ms = (time.perf_counter() - t0) * 1000.0

    label = "Positive" if p >= threshold else "Negative"
    conf = sigmoid_confidence(p)

    st.subheader("Prediction")
    m1, m2, m3 = st.columns(3)
    with m1:
        st.metric("Label", label)
    with m2:
        st.metric("P(positive)", f"{p:.3f}")
    with m3:
        st.metric("Confidence", f"{conf:.3f}")

    st.write("Probability bar")
    st.progress(min(max(p, 0.0), 1.0))

    if show_latency:
        st.caption(f"Inference latency: {dt_ms:.1f} ms")
