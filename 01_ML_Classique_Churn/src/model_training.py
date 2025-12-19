from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV

import xgboost as xgb


@dataclass
class TrainedModels:
    models: Dict[str, object]


class ChurnModelTrainer:
    def __init__(self, X_train: np.ndarray, y_train: pd.Series):
        self.X_train = X_train
        self.y_train = y_train
        self.models: Dict[str, object] = {}

    def train_random_forest(self) -> RandomForestClassifier:
        param_grid = {
            "n_estimators": [200],
            "max_depth": [10, 15],
            "min_samples_split": [5, 10],
            "random_state": [42],
        }

        rf = RandomForestClassifier()
        grid_search = GridSearchCV(
            rf,
            param_grid,
            cv=5,
            scoring="roc_auc",
            n_jobs=-1,
            verbose=0,
        )
        grid_search.fit(self.X_train, self.y_train)

        self.models["RandomForest"] = grid_search.best_estimator_
        return grid_search.best_estimator_

    def train_xgboost(self) -> xgb.XGBClassifier:
        param_grid = {
            "n_estimators": [200],
            "max_depth": [5, 7],
            "learning_rate": [0.05, 0.1],
            "subsample": [0.8, 1.0],
        }

        xgb_model = xgb.XGBClassifier(
            random_state=42,
            eval_metric="logloss",
        )

        grid_search = GridSearchCV(
            xgb_model,
            param_grid,
            cv=5,
            scoring="roc_auc",
            n_jobs=-1,
            verbose=0,
        )
        grid_search.fit(self.X_train, self.y_train)

        self.models["XGBoost"] = grid_search.best_estimator_
        return grid_search.best_estimator_

    def train_both(self) -> TrainedModels:
        self.train_random_forest()
        self.train_xgboost()
        return TrainedModels(models=self.models)
