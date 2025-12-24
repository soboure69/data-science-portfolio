import os
import unittest

import pandas as pd

from src.recommendation_engine import RecommendationEngine


class TestRecommendationEngine(unittest.TestCase):
    def setUp(self):
        project_dir = os.path.dirname(os.path.dirname(__file__))
        data_path = os.path.join(project_dir, "data", "products.csv")
        self.engine = RecommendationEngine(data_path=data_path)

    def test_load_products_has_required_columns(self):
        df = self.engine.load_products()
        self.assertIsInstance(df, pd.DataFrame)
        self.assertGreater(len(df), 0)
        for col in [
            "product_id",
            "name",
            "category",
            "price",
            "rating",
            "num_reviews",
            "description",
            "tags",
        ]:
            self.assertIn(col, df.columns)

    def test_build_features_creates_similarity_matrix(self):
        self.engine.load_products()
        self.engine.build_features()
        self.assertIsNotNone(self.engine._sim_matrix)
        self.assertEqual(self.engine._sim_matrix.shape[0], len(self.engine.products))
        self.assertEqual(self.engine._sim_matrix.shape[1], len(self.engine.products))

    def test_recommend_returns_results(self):
        df = self.engine.load_products()
        self.engine.build_features()
        product_id = df.iloc[0]["product_id"]
        recs = self.engine.recommend(product_id, n_recommendations=5)
        self.assertLessEqual(len(recs), 5)
        for r in recs:
            self.assertNotEqual(r.product_id, product_id)
            self.assertGreaterEqual(r.score, 0.0)

    def test_recommend_category_filter(self):
        df = self.engine.load_products()
        self.engine.build_features()
        product_id = df.iloc[0]["product_id"]
        category = df.iloc[0]["category"]
        recs = self.engine.recommend(product_id, n_recommendations=10, category=category)
        if recs:
            rec_ids = [r.product_id for r in recs]
            cats = df[df["product_id"].isin(rec_ids)]["category"].unique().tolist()
            self.assertEqual(set(cats), {category})

    def test_recommend_unknown_product_raises(self):
        self.engine.load_products()
        self.engine.build_features()
        with self.assertRaises(ValueError):
            self.engine.recommend("UNKNOWN", n_recommendations=5)


if __name__ == "__main__":
    unittest.main()
