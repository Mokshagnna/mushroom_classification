from __future__ import annotations

import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import pandas as pd

from src.mlops_pipeline.monitoring_ops import (
    api_performance,
    categorical_drift,
    prediction_drift,
)
from src.mushroom_classification.validation import FEATURE_COLUMNS


class MushroomMonitoringOpsTests(unittest.TestCase):
    def test_categorical_drift_uses_mushroom_feature_schema(self) -> None:
        root = Path(__file__).resolve().parents[1]
        frame = pd.read_csv(root / "data" / "raw" / "mushroom.csv")
        baseline = frame.loc[:, list(FEATURE_COLUMNS)].head(100)
        changed = baseline.copy()
        changed["habitat"] = "w"

        report = categorical_drift(baseline, changed)

        self.assertEqual(report["feature_count"], 22)
        self.assertGreater(report["feature_scores"]["habitat"]["total_variation"], 0)
        self.assertFalse(report["ground_truth_required"])

    def test_prediction_drift_compares_distribution_without_accuracy_claim(self) -> None:
        with TemporaryDirectory() as directory:
            baseline = Path(directory) / "baseline.jsonl"
            current = Path(directory) / "current.jsonl"
            baseline.write_text(
                json.dumps(
                    {
                        "status": "success",
                        "prediction": 0,
                        "poisonous_probability": 0.1,
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            current.write_text(
                json.dumps(
                    {
                        "status": "success",
                        "prediction": 1,
                        "poisonous_probability": 0.9,
                    }
                )
                + "\n",
                encoding="utf-8",
            )

            report = prediction_drift(baseline, current)

            self.assertEqual(report["total_variation"], 1.0)
            self.assertIsNone(report["accuracy"])

    def test_api_performance_reports_latency_not_accuracy(self) -> None:
        with TemporaryDirectory() as directory:
            config = {
                "api": {"monitoring_log": str(Path(directory) / "events.jsonl")}
            }
            Path(config["api"]["monitoring_log"]).write_text(
                json.dumps(
                    {
                        "status": "success",
                        "prediction": 1,
                        "poisonous_probability": 0.8,
                        "latency_ms": 4.0,
                    }
                )
                + "\n",
                encoding="utf-8",
            )

            report = api_performance(config)

            self.assertEqual(report["successful_predictions"], 1)
            self.assertEqual(report["latency_ms"]["p95"], 4.0)
            self.assertIsNone(report["accuracy"])


if __name__ == "__main__":
    unittest.main()
