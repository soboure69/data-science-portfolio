from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict

import matplotlib

# Non-interactive backend to avoid RecursionError in some Windows/VS Code setups
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    ConfusionMatrixDisplay,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


@dataclass
class ModelMetrics:
    accuracy: float
    precision: float
    recall: float
    f1: float
    roc_auc: float
    confusion_matrix: list[list[int]]


class ChurnModelEvaluator:
    def __init__(self, models: Dict[str, object], X_test: np.ndarray, y_test: pd.Series):
        self.models = models
        self.X_test = X_test
        self.y_test = y_test
        self.results: Dict[str, ModelMetrics] = {}

    def evaluate_all(self) -> Dict[str, ModelMetrics]:
        for name, model in self.models.items():
            y_pred = model.predict(self.X_test)
            y_pred_proba = model.predict_proba(self.X_test)[:, 1]

            self.results[name] = ModelMetrics(
                accuracy=float(accuracy_score(self.y_test, y_pred)),
                precision=float(precision_score(self.y_test, y_pred)),
                recall=float(recall_score(self.y_test, y_pred)),
                f1=float(f1_score(self.y_test, y_pred)),
                roc_auc=float(roc_auc_score(self.y_test, y_pred_proba)),
                confusion_matrix=confusion_matrix(self.y_test, y_pred).tolist(),
            )

        return self.results

    def save_metrics_json(self, out_path: str | Path) -> None:
        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)

        serializable = {
            name: {
                "accuracy": m.accuracy,
                "precision": m.precision,
                "recall": m.recall,
                "f1": m.f1,
                "roc_auc": m.roc_auc,
                "confusion_matrix": m.confusion_matrix,
            }
            for name, m in self.results.items()
        }

        out_path.write_text(json.dumps(serializable, indent=2), encoding="utf-8")

    def plot_confusion_matrices(self, out_path: str | Path) -> None:
        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)

        n = len(self.models)
        fig, axes = plt.subplots(1, n, figsize=(6 * n, 4))
        if n == 1:
            axes = [axes]

        for ax, (name, model) in zip(axes, self.models.items(), strict=False):
            y_pred = model.predict(self.X_test)
            cm = confusion_matrix(self.y_test, y_pred)

            # seaborn.heatmap can trigger RecursionError in some Windows/matplotlib setups.
            # ConfusionMatrixDisplay is simpler and more robust.
            disp = ConfusionMatrixDisplay(confusion_matrix=cm)
            disp.plot(ax=ax, colorbar=False, values_format="d")
            ax.set_title(f"{name} - Confusion Matrix")

        plt.tight_layout()
        fig.savefig(out_path, dpi=200, bbox_inches="tight")
        plt.close(fig)
