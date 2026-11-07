import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import joblib
import numpy as np
import pandas as pd

from src.mushroom_classification.evaluation import calculate_metrics
from src.mushroom_classification.inference import predict_frame
from src.mushroom_classification.models import build_model_pipeline
from src.mushroom_classification.pipeline import save_model_with_backup
from src.mushroom_classification.validation import (
    FEATURE_COLUMNS,
    TARGET_COLUMN,
    encode_target,
    normalize_features,
    validate_dataset_schema,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]


class MushroomValidationTests(unittest.TestCase):
    def test_shipped_dataset_schema_and_unknown_marker(self):
        dataset = pd.read_csv(PROJECT_ROOT / "data" / "raw" / "mushroom.csv")

        report = validate_dataset_schema(dataset)

        self.assertTrue(report["valid"], report["issues"])
        self.assertEqual(report["row_count"], 8124)
        self.assertEqual(report["class_distribution"], {"e": 4208, "p": 3916})
        self.assertEqual(report["unknown_stalk_root_values"], 2480)
        self.assertEqual(report["constant_features"], ["veil-type"])

    def test_invalid_target_is_rejected(self):
        dataset = pd.read_csv(PROJECT_ROOT / "data" / "raw" / "mushroom.csv").head(3)
        dataset.loc[dataset.index[0], TARGET_COLUMN] = "unknown"

        report = validate_dataset_schema(dataset)

        self.assertFalse(report["valid"])
        self.assertIn("unknown", report["invalid_targets"])

    def test_invalid_feature_category_is_rejected(self):
        dataset = pd.read_csv(PROJECT_ROOT / "data" / "raw" / "mushroom.csv").head(3)
        dataset.loc[dataset.index[0], "odor"] = "customer"

        report = validate_dataset_schema(dataset)

        self.assertFalse(report["valid"])
        self.assertEqual(report["invalid_categories"]["odor"], ["customer"])

    def test_target_mapping_marks_poisonous_as_positive(self):
        target = pd.Series(["e", "p", "e", "p"])

        self.assertEqual(encode_target(target).tolist(), [0, 1, 0, 1])

    def test_confusion_matrix_labels_and_poisonous_recall(self):
        metrics = calculate_metrics(
            np.asarray([0, 0, 1, 1]),
            np.asarray([0, 1, 1, 1]),
            np.asarray([0.1, 0.6, 0.7, 0.9]),
        )

        self.assertEqual(metrics["confusion_matrix"], [[1, 1], [0, 2]])
        self.assertEqual(metrics["confusion_matrix_labels"], ["edible", "poisonous"])
        self.assertEqual(metrics["poisonous_recall"], 1.0)
        self.assertEqual(metrics["edible_recall"], 0.5)

    def test_pipeline_handles_missing_and_unseen_categories(self):
        dataset = pd.read_csv(PROJECT_ROOT / "data" / "raw" / "mushroom.csv").head(100)
        x_train = normalize_features(dataset)
        y_train = encode_target(dataset[TARGET_COLUMN])
        estimator = build_model_pipeline("Decision Tree", random_state=42, n_estimators=20)
        estimator.fit(x_train, y_train)

        inference_row = dataset.loc[:, list(FEATURE_COLUMNS)].iloc[[0]].copy()
        inference_row.loc[:, "cap-color"] = "__unseen_category__"
        inference_row.loc[:, "stalk-root"] = "?"
        predictions = predict_frame(estimator, inference_row)

        self.assertEqual(len(predictions), 1)
        self.assertIn(predictions.loc[0, "predicted_class"], {"edible", "poisonous"})
        self.assertAlmostEqual(
            predictions.loc[0, "edible_probability"]
            + predictions.loc[0, "poisonous_probability"],
            1.0,
        )

    def test_inference_rejects_missing_feature_columns(self):
        with self.assertRaisesRegex(ValueError, "Missing expected feature columns"):
            predict_frame(None, pd.DataFrame({"cap-shape": ["x"]}))

    def test_existing_model_is_backed_up_before_replacement(self):
        with TemporaryDirectory() as directory:
            model_path = Path(directory) / "model.joblib"
            joblib.dump({"version": 1}, model_path)

            backup_path = save_model_with_backup({"version": 2}, model_path)

            self.assertIsNotNone(backup_path)
            self.assertEqual(joblib.load(backup_path), {"version": 1})
            self.assertEqual(joblib.load(model_path), {"version": 2})


if __name__ == "__main__":
    unittest.main()
