from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

from .config import get_model_path
from .evaluation import CLASS_NAMES
from .validation import TARGET_COLUMN, normalize_features, validate_inference_features


def predict_frame(estimator: Any, records: pd.DataFrame) -> pd.DataFrame:
    issues = validate_inference_features(records)
    if issues:
        raise ValueError("Invalid inference data: " + "; ".join(issues))

    features = normalize_features(records)
    predictions = np.asarray(estimator.predict(features), dtype=int)
    probabilities = np.asarray(estimator.predict_proba(features))
    classes = list(estimator.classes_)
    edible_probability = probabilities[:, classes.index(0)]
    poisonous_probability = probabilities[:, classes.index(1)]

    result = records.copy()
    if TARGET_COLUMN in result.columns:
        result = result.drop(columns=[TARGET_COLUMN])
    result["predicted_class"] = [CLASS_NAMES[label] for label in predictions]
    result["edible_probability"] = edible_probability
    result["poisonous_probability"] = poisonous_probability
    return result


def run_inference(input_path: Path, output_path: Path) -> Path:
    model_path = get_model_path()
    if not model_path.is_file():
        raise FileNotFoundError(f"Trained model not found: {model_path}. Run the training pipeline first.")
    if not input_path.is_file():
        raise FileNotFoundError(f"Inference input not found: {input_path}")

    estimator = joblib.load(model_path)
    predictions = predict_frame(estimator, pd.read_csv(input_path))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    predictions.to_csv(output_path, index=False)
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Run mushroom-classification batch inference.")
    parser.add_argument("--input", type=Path, required=True, help="CSV file with mushroom features.")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs/inference_predictions.csv"),
        help="Destination CSV for predictions.",
    )
    arguments = parser.parse_args()
    output_path = run_inference(arguments.input, arguments.output)
    print(f"Predictions saved to {output_path}")
    print(
        "Safety notice: this academic model is not a food-safety tool. "
        "Never use it to decide whether a mushroom is safe to eat."
    )
