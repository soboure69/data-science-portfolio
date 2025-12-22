from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Tuple

import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds


@dataclass
class TextDataset:
    X_train: np.ndarray
    y_train: np.ndarray
    X_val: np.ndarray
    y_val: np.ndarray
    X_test: np.ndarray
    y_test: np.ndarray
    vectorizer: tf.keras.layers.TextVectorization
    word_index: dict[str, int]
    max_words: int
    max_len: int


class TextPreprocessor:
    def __init__(self, max_words: int = 10_000, max_len: int = 200):
        self.max_words = max_words
        self.max_len = max_len

    @staticmethod
    def _custom_standardize(text: tf.Tensor) -> tf.Tensor:
        text = tf.strings.lower(text)
        text = tf.strings.regex_replace(text, r"<br\s*/?>", " ")
        text = tf.strings.regex_replace(text, r"[^a-z0-9\s]", "")
        text = tf.strings.regex_replace(text, r"\s+", " ")
        return tf.strings.strip(text)

    def build_vectorizer_model(self, vectorizer: tf.keras.layers.TextVectorization) -> tf.keras.Model:
        model = tf.keras.Sequential([vectorizer])
        model(tf.constant(["warmup"]))
        return model

    def _adapt_vectorizer(self, texts: Iterable[str]) -> tf.keras.layers.TextVectorization:
        vectorizer = tf.keras.layers.TextVectorization(
            max_tokens=self.max_words,
            output_mode="int",
            output_sequence_length=self.max_len,
            standardize=self._custom_standardize,
        )
        vectorizer.adapt(texts)
        return vectorizer

    def load_imdb_text(self, validation_size: int = 5000, seed: int = 42) -> TextDataset:
        (train_ds, test_ds), _ = tfds.load(
            "imdb_reviews",
            split=["train", "test"],
            as_supervised=True,
            with_info=True,
        )

        train_texts = train_ds.map(lambda x, y: x)
        vectorizer = self._adapt_vectorizer(train_texts.batch(1024))
        vocab = vectorizer.get_vocabulary()
        word_index = {w: i for i, w in enumerate(vocab)}

        def to_xy(ds: tf.data.Dataset) -> Tuple[np.ndarray, np.ndarray]:
            xs = []
            ys = []
            for x, y in tfds.as_numpy(ds):
                xs.append(x.decode("utf-8"))
                ys.append(int(y))
            x_vec = vectorizer(tf.constant(xs)).numpy()
            y_arr = np.asarray(ys, dtype=np.int32)
            return x_vec, y_arr

        X_all, y_all = to_xy(train_ds)
        rng = np.random.default_rng(seed)
        idx = rng.permutation(len(X_all))
        X_all = X_all[idx]
        y_all = y_all[idx]

        X_val = X_all[:validation_size]
        y_val = y_all[:validation_size]
        X_train = X_all[validation_size:]
        y_train = y_all[validation_size:]

        X_test, y_test = to_xy(test_ds)

        return TextDataset(
            X_train=X_train,
            y_train=y_train,
            X_val=X_val,
            y_val=y_val,
            X_test=X_test,
            y_test=y_test,
            vectorizer=vectorizer,
            word_index=word_index,
            max_words=self.max_words,
            max_len=self.max_len,
        )
