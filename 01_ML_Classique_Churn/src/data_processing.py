from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler


@dataclass
class PreparedData:
    X_train: np.ndarray
    X_test: np.ndarray
    y_train: pd.Series
    y_test: pd.Series
    scaler: StandardScaler
    label_encoders: Dict[str, LabelEncoder]
    feature_names: list[str]


class ChurnDataProcessor:
    def __init__(self, filepath: str | Path):
        self.filepath = Path(filepath)
        self.df: pd.DataFrame | None = None
        self.label_encoders: Dict[str, LabelEncoder] = {}

    def load(self) -> pd.DataFrame:
        self.df = pd.read_csv(self.filepath)
        return self.df

    def clean(self) -> pd.DataFrame:
        if self.df is None:
            raise ValueError("Dataframe not loaded. Call load() first.")

        df = self.df.copy()

        if "customerID" in df.columns:
            df = df.drop(["customerID"], axis=1)

        if "Churn" not in df.columns:
            raise ValueError("Expected target column 'Churn' in dataset.")

        df["Churn"] = (df["Churn"] == "Yes").astype(int)

        if "TotalCharges" in df.columns:
            df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
            df["TotalCharges"] = df["TotalCharges"].fillna(df["TotalCharges"].median())

        self.df = df
        return self.df

    def feature_engineering(self) -> pd.DataFrame:
        if self.df is None:
            raise ValueError("Dataframe not loaded. Call load() first.")

        df = self.df.copy()

        if "tenure" in df.columns:
            df["tenure_group"] = pd.cut(
                df["tenure"],
                bins=[0, 12, 24, 48, 72],
                labels=["0-1 ans", "1-2 ans", "2-4 ans", "4+ ans"],
                include_lowest=True,
            )

        if "MonthlyCharges" in df.columns and "TotalCharges" in df.columns:
            df["monthly_to_total_ratio"] = df["MonthlyCharges"] / (df["TotalCharges"] + 1)

        self.df = df
        return self.df

    def encode_categorical(self) -> pd.DataFrame:
        if self.df is None:
            raise ValueError("Dataframe not loaded. Call load() first.")

        df = self.df.copy()
        categorical_cols = df.select_dtypes(include=["object", "category"]).columns

        for col in categorical_cols:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col].astype(str))
            self.label_encoders[col] = le

        self.df = df
        return self.df

    def prepare_for_modeling(self, test_size: float = 0.2, random_state: int = 42) -> PreparedData:
        if self.df is None:
            raise ValueError("Dataframe not loaded. Call load() first.")

        df = self.df.copy()
        X = df.drop("Churn", axis=1)
        y = df["Churn"]

        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=test_size,
            random_state=random_state,
            stratify=y,
        )

        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        return PreparedData(
            X_train=X_train_scaled,
            X_test=X_test_scaled,
            y_train=y_train,
            y_test=y_test,
            scaler=scaler,
            label_encoders=self.label_encoders,
            feature_names=list(X.columns),
        )
