from __future__ import annotations

import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from src.mushroom_classification.monitoring import PredictionMonitor


class PredictionMonitorTests(unittest.TestCase):
    def test_records_prediction_distribution_and_latency_without_accuracy(self) -> None:
        with TemporaryDirectory() as directory:
            path = Path(directory) / "nested" / "predictions.jsonl"
            monitor = PredictionMonitor(path)
            monitor.record_prediction(0, 0.1, 2.5, request_id="edible-request")
            monitor.record_prediction(1, 0.9, 3.5, request_id="poisonous-request")
            monitor.record_failure("request_validation", request_id="invalid-request")

            summary = monitor.summary()
            self.assertEqual(summary["total_predictions"], 2)
            self.assertEqual(summary["failed_requests"], 1)
            self.assertEqual(summary["prediction_counts"], {"edible": 1, "poisonous": 1})
            self.assertAlmostEqual(summary["mean_poisonous_probability"], 0.5)
            self.assertAlmostEqual(summary["mean_latency_ms"], 3.0)
            self.assertEqual(summary["max_latency_ms"], 3.5)
            self.assertIsNone(summary["accuracy"])

            events = [json.loads(line) for line in path.read_text().splitlines()]
            self.assertNotIn("input", events[0])

    def test_rejects_invalid_prediction_statistics(self) -> None:
        with TemporaryDirectory() as directory:
            monitor = PredictionMonitor(Path(directory) / "predictions.jsonl")

            with self.assertRaises(ValueError):
                monitor.record_prediction(2, 0.5, 1.0)
            with self.assertRaises(ValueError):
                monitor.record_prediction(0, 1.1, 1.0)
            with self.assertRaises(ValueError):
                monitor.record_prediction(0, 0.5, -0.1)

    def test_reports_malformed_persisted_event(self) -> None:
        with TemporaryDirectory() as directory:
            path = Path(directory) / "predictions.jsonl"
            path.write_text("not-json\n", encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "line 1"):
                PredictionMonitor(path).summary()


if __name__ == "__main__":
    unittest.main()
