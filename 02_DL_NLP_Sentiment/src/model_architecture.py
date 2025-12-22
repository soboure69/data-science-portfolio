from __future__ import annotations

from dataclasses import dataclass

import tensorflow as tf


@dataclass
class ModelConfig:
    vocab_size: int = 10_000
    max_len: int = 200
    embedding_dim: int = 128
    rnn_units: int = 64
    dropout: float = 0.3


def build_bilstm_model(cfg: ModelConfig) -> tf.keras.Model:
    model = tf.keras.Sequential(
        [
            tf.keras.layers.Embedding(cfg.vocab_size, cfg.embedding_dim, input_length=cfg.max_len),
            tf.keras.layers.SpatialDropout1D(cfg.dropout),
            tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(cfg.rnn_units, dropout=cfg.dropout)),
            tf.keras.layers.Dense(1, activation="sigmoid"),
        ]
    )
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss="binary_crossentropy",
        metrics=["accuracy", tf.keras.metrics.AUC(name="roc_auc")],
    )
    return model


def build_gru_model(cfg: ModelConfig) -> tf.keras.Model:
    model = tf.keras.Sequential(
        [
            tf.keras.layers.Embedding(cfg.vocab_size, cfg.embedding_dim, input_length=cfg.max_len),
            tf.keras.layers.SpatialDropout1D(cfg.dropout),
            tf.keras.layers.Bidirectional(tf.keras.layers.GRU(cfg.rnn_units, dropout=cfg.dropout)),
            tf.keras.layers.Dense(1, activation="sigmoid"),
        ]
    )
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss="binary_crossentropy",
        metrics=["accuracy", tf.keras.metrics.AUC(name="roc_auc")],
    )
    return model
