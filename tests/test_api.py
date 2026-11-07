from __future__ import annotations

import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import pandas as pd
from fastapi.testclient import TestClient

from api.main import create_app, load_model
from src.mushroom_classification.monitoring import PredictionMonitor
from src.mushroom_classification.validation import FEATURE_COLUMNS

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = PROJECT_ROOT / "models" / "mushroom_classifier.joblib"


class MushroomApiTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = TemporaryDirectory()
        monitor = PredictionMonitor(Path(self.temp_dir.name) / "monitor.jsonl")
        app = create_app(model_path=MODEL_PATH, monitor=monitor)
        self.client_context = TestClient(app)
        self.client = self.client_context.__enter__()
        self.addCleanup(self.client_context.__exit__, None, None, None)
        self.addCleanup(self.temp_dir.cleanup)
        self.features = (
            pd.read_csv(PROJECT_ROOT / "data" / "raw" / "mushroom.csv")
            .loc[0, list(FEATURE_COLUMNS)]
            .to_dict()
        )

    def test_health_endpoint_and_model_startup(self) -> None:
        response = self.client.get("/health")

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json(), {"status": "healthy", "model_loaded": True})
        self.assertEqual(self.client.get("/ready").json(), {"status": "ready"})

    def test_loads_saved_mushroom_pipeline(self) -> None:
        model = load_model(MODEL_PATH)

        self.assertEqual(set(model.classes_), {0, 1})
        with self.assertRaises(FileNotFoundError):
            load_model(Path(self.temp_dir.name) / "missing.joblib")

    def test_valid_mushroom_prediction_uses_edible_poisonous_labels(self) -> None:
        response = self.client.post("/predict", json=self.features)

        self.assertEqual(response.status_code, 200, response.text)
        result = response.json()
        self.assertIn(result["prediction"], (0, 1))
        self.assertEqual(
            result["predicted_class"], "poisonous" if result["prediction"] == 1 else "edible"
        )
        self.assertGreaterEqual(result["edible_probability"], 0)
        self.assertGreaterEqual(result["poisonous_probability"], 0)
        self.assertAlmostEqual(
            result["edible_probability"] + result["poisonous_probability"], 1.0
        )

    def test_unknown_stalk_root_marker_uses_training_preprocessing(self) -> None:
        features = dict(self.features, **{"stalk-root": "?"})

        response = self.client.post("/predict", json=features)

        self.assertEqual(response.status_code, 200, response.text)

    def test_batch_prediction_returns_mushroom_results(self) -> None:
        response = self.client.post(
            "/predict/batch", json={"records": [self.features, self.features]}
        )

        self.assertEqual(response.status_code, 200, response.text)
        result = response.json()
        self.assertEqual(result["count"], 2)
        self.assertEqual(len(result["predictions"]), 2)
        self.assertTrue(
            all(item["predicted_class"] in {"edible", "poisonous"} for item in result["predictions"])
        )

    def test_invalid_or_customer_shaped_payload_is_rejected(self) -> None:
        invalid_features = dict(self.features)
        invalid_features["odor"] = "customer"

        response = self.client.post("/predict", json=invalid_features)
        self.assertEqual(response.status_code, 422)
        self.assertEqual(response.json()["message"], "Request validation failed.")

        customer_payload = dict(self.features, customerID="not-a-mushroom-feature")
        self.assertEqual(self.client.post("/predict", json=customer_payload).status_code, 422)

    def test_prediction_monitoring_does_not_store_features_or_claim_accuracy(self) -> None:
        self.client.post("/predict", json=self.features)
        monitoring_path = Path(self.temp_dir.name) / "monitor.jsonl"
        raw_log = monitoring_path.read_text(encoding="utf-8")
        summary = self.client.app.state.monitor.summary()

        self.assertNotIn("cap-shape", raw_log)
        self.assertEqual(summary["total_predictions"], 1)
        self.assertEqual(summary["accuracy"], None)
        self.assertIn("ground-truth", summary["accuracy_note"])


if __name__ == "__main__":
    unittest.main()
