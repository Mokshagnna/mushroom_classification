"""Compatibility imports for mushroom dataset validation."""

import json
from pathlib import Path
from typing import Any

import pandas as pd

from src.mushroom_classification.validation import (
    EXPECTED_COLUMNS,
    TARGET_COLUMN,
    TARGET_LABELS,
    validate_dataset_schema,
)

TARGET_VALUES = set(TARGET_LABELS)


def validate_preprocessed_data(
    x_train: pd.DataFrame,
    x_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
) -> dict[str, Any]:
    issues = []
    if x_train.empty or x_test.empty:
        issues.append("Train/test feature matrices are empty.")
    if len(x_train) != len(y_train):
        issues.append("Training feature and target lengths do not match.")
    if len(x_test) != len(y_test):
        issues.append("Testing feature and target lengths do not match.")
    if not x_train.columns.equals(x_test.columns):
        issues.append("Train and test feature columns differ after encoding.")
    if x_train.isnull().values.any() or x_test.isnull().values.any():
        issues.append("Missing values detected in encoded features.")
    if y_train.isnull().any() or y_test.isnull().any():
        issues.append("Missing values detected in target vectors.")
    invalid_targets = set(y_train.unique()) | set(y_test.unique())
    if not invalid_targets.issubset({0, 1}):
        issues.append(f"Target vectors contain values outside {{0, 1}}: {sorted(invalid_targets)}")

    return {
        "valid": not issues,
        "train_shape": list(x_train.shape),
        "test_shape": list(x_test.shape),
        "train_target_distribution": {
            str(key): int(value) for key, value in y_train.value_counts().items()
        },
        "test_target_distribution": {
            str(key): int(value) for key, value in y_test.value_counts().items()
        },
        "feature_count": int(x_train.shape[1]),
        "issues": issues,
    }


def save_validation_report(report: dict[str, Any], output_path: str) -> str:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=4), encoding="utf-8")
    return str(path)
