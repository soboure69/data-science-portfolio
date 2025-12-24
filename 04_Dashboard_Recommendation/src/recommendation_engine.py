import os
from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import StandardScaler
from scipy.sparse import hstack, csr_matrix


@dataclass(frozen=True)
class RecommendationResult:
    product_id: str
    score: float


class RecommendationEngine:
    def __init__(
        self,
        data_path: Optional[str] = None,
        text_weight: float = 0.85,
        numeric_weight: float = 0.15,
        min_df: int = 1,
    ):
        self.data_path = data_path or os.path.join(
            os.path.dirname(os.path.dirname(__file__)), "data", "products.csv"
        )
        self.text_weight = float(text_weight)
        self.numeric_weight = float(numeric_weight)
        self.min_df = int(min_df)

        self.products: Optional[pd.DataFrame] = None
        self._tfidf: Optional[TfidfVectorizer] = None
        self._scaler: Optional[StandardScaler] = None
        self._feature_matrix = None
        self._sim_matrix: Optional[np.ndarray] = None
        self._id_to_idx: dict[str, int] = {}
        self._idx_to_id: list[str] = []

    def load_products(self) -> pd.DataFrame:
        df = pd.read_csv(self.data_path)

        required = {
            "product_id",
            "name",
            "category",
            "price",
            "rating",
            "num_reviews",
            "description",
            "tags",
        }
        missing = required - set(df.columns)
        if missing:
            raise ValueError(f"Missing columns in products.csv: {sorted(missing)}")

        df = df.copy()
        df["product_id"] = df["product_id"].astype(str)
        df["category"] = df["category"].astype(str)
        df["name"] = df["name"].astype(str)
        df["description"] = df["description"].astype(str)
        df["tags"] = df["tags"].astype(str)

        df["price"] = pd.to_numeric(df["price"], errors="coerce")
        df["rating"] = pd.to_numeric(df["rating"], errors="coerce")
        df["num_reviews"] = pd.to_numeric(df["num_reviews"], errors="coerce")

        df = df.dropna(subset=["product_id", "name", "category", "price", "rating", "num_reviews"])

        self.products = df.reset_index(drop=True)
        self._id_to_idx = {pid: i for i, pid in enumerate(self.products["product_id"].tolist())}
        self._idx_to_id = self.products["product_id"].tolist()
        return self.products

    def build_features(self) -> None:
        if self.products is None:
            self.load_products()

        df = self.products
        text = (
            df["name"].fillna("")
            + " "
            + df["category"].fillna("")
            + " "
            + df["description"].fillna("")
            + " "
            + df["tags"].fillna("")
        )

        self._tfidf = TfidfVectorizer(
            lowercase=True,
            stop_words="english",
            min_df=self.min_df,
            ngram_range=(1, 2),
        )
        X_text = self._tfidf.fit_transform(text)

        numeric = df[["price", "rating", "num_reviews"]].to_numpy(dtype=float)
        self._scaler = StandardScaler(with_mean=True, with_std=True)
        X_num = self._scaler.fit_transform(numeric)
        X_num = csr_matrix(X_num)

        X = hstack(
            [X_text.multiply(self.text_weight), X_num.multiply(self.numeric_weight)],
            format="csr",
        )

        self._feature_matrix = X
        self._sim_matrix = cosine_similarity(X, X)

    def recommend(
        self,
        product_id: str,
        n_recommendations: int = 5,
        category: Optional[str] = None,
        max_price: Optional[float] = None,
    ) -> list[RecommendationResult]:
        if self.products is None:
            self.load_products()
        if self._sim_matrix is None:
            self.build_features()

        if product_id not in self._id_to_idx:
            raise ValueError(f"Unknown product_id: {product_id}")

        idx = self._id_to_idx[product_id]
        scores = self._sim_matrix[idx].copy()
        scores[idx] = -1.0

        df = self.products
        mask = np.ones(len(df), dtype=bool)
        if category and category != "All":
            mask &= df["category"].astype(str).eq(str(category))
        if max_price is not None and np.isfinite(max_price):
            mask &= df["price"].astype(float).le(float(max_price))

        candidate_idx = np.where(mask)[0]
        if len(candidate_idx) == 0:
            return []

        ordered = candidate_idx[np.argsort(scores[candidate_idx])[::-1]]
        ordered = [i for i in ordered if scores[i] > 0]

        results: list[RecommendationResult] = []
        for i in ordered:
            if len(results) >= int(n_recommendations):
                break
            results.append(RecommendationResult(product_id=self._idx_to_id[i], score=float(scores[i])))

        return results

    def get_product(self, product_id: str) -> dict:
        if self.products is None:
            self.load_products()
        row = self.products[self.products["product_id"] == product_id]
        if row.empty:
            raise ValueError(f"Unknown product_id: {product_id}")
        return row.iloc[0].to_dict()
