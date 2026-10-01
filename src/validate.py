"""Schema and preprocessing validation for the Mushroom Classification project."""

import json
import os
from typing import Any, Dict, List

import pandas as pd

EXPECTED_COLUMNS = [
    "class",
    "cap-shape",
    "cap-surface",
    "cap-color",
    "bruises",
    "odor",
    "gill-attachment",
    "gill-spacing",
    "gill-size",
    "gill-color",
    "stalk-shape",
    "stalk-root",
    "stalk-surface-above-ring",
    "stalk-surface-below-ring",
    "stalk-color-above-ring",
    "stalk-color-below-ring",
    "veil-type",
    "veil-color",
    "ring-number",
    "ring-type",
    "spore-print-color",
    "population",
    "habitat",
]

TARGET_COLUMN = "class"
TARGET_VALUES = {"e", "p"}


def validate_dataset_schema(df: pd.DataFrame) -> Dict[str, Any]:
    """Validate the raw dataset against the known mushroom schema."""
    issues: List[str] = []

    if df.empty:
        issues.append("Dataset is empty.")

    missing_columns = [col for col in EXPECTED_COLUMNS if col not in df.columns]
    if missing_columns:
        issues.append(f"Missing expected columns: {missing_columns}")

    extra_columns = [col for col in df.columns if col not in EXPECTED_COLUMNS]
    if extra_columns:
        issues.append(f"Unexpected columns detected: {extra_columns}")

    if TARGET_COLUMN in df.columns:
        invalid_targets = df.loc[~df[TARGET_COLUMN].isin(TARGET_VALUES), TARGET_COLUMN]
        if not invalid_targets.empty:
            issues.append(f"Found invalid target values: {sorted(set(invalid_targets.tolist()))}")
    else:
        issues.append(f"Target column '{TARGET_COLUMN}' is missing.")

    missing_values = df.isnull().sum()
    missing_cols = [col for col, count in missing_values.items() if count > 0]
    if missing_cols:
        issues.append(f"Missing values detected in columns: {missing_cols}")

    duplicate_rows = int(df.duplicated().sum())
    if duplicate_rows > 0:
        issues.append(f"Duplicate rows detected: {duplicate_rows}")

    constant_features = [col for col in df.columns if df[col].nunique() <= 1]
    if constant_features:
        # This is expected for a feature like veil-type, but it is tracked for traceability.
        pass

    return {
        "valid": len(issues) == 0,
        "expected_columns": EXPECTED_COLUMNS,
        "missing_columns": missing_columns,
        "extra_columns": extra_columns,
        "row_count": int(df.shape[0]),
        "column_count": int(df.shape[1]),
        "missing_values": {str(col): int(count) for col, count in missing_values.items()},
        "duplicate_rows": duplicate_rows,
        "constant_features": constant_features,
        "issues": issues,
    }


def validate_preprocessed_data(X_train: pd.DataFrame, X_test: pd.DataFrame, y_train: pd.Series, y_test: pd.Series) -> Dict[str, Any]:
    """Validate the encoded feature matrix and target vectors."""
    issues: List[str] = []

    if X_train.empty or X_test.empty:
        issues.append("Train/test feature matrices are empty.")

    if len(X_train) != len(y_train):
        issues.append("Training feature and target lengths do not match.")
    if len(X_test) != len(y_test):
        issues.append("Testing feature and target lengths do not match.")

    if not X_train.columns.equals(X_test.columns):
        issues.append("Train and test feature columns differ after encoding.")

    if X_train.isnull().values.any() or X_test.isnull().values.any():
        issues.append("Missing values detected in encoded features.")

    if y_train.isnull().any() or y_test.isnull().any():
        issues.append("Missing values detected in target vectors.")

    invalid_targets = set(y_train.unique()) | set(y_test.unique())
    if not invalid_targets.issubset({0, 1}):
        issues.append(f"Target vectors contain values outside {0, 1}: {sorted(map(int, invalid_targets))}")

    valid_numeric_columns = X_train.select_dtypes(include=["number", "bool"]).columns.tolist()
    remaining_train = [col for col in X_train.columns if col not in valid_numeric_columns]
    valid_numeric_columns_test = X_test.select_dtypes(include=["number", "bool"]).columns.tolist()
    remaining_test = [col for col in X_test.columns if col not in valid_numeric_columns_test]
    non_numeric = remaining_train + remaining_test
    if non_numeric:
        issues.append(f"Non-numeric feature columns remain after encoding: {sorted(set(non_numeric))}")

    return {
        "valid": len(issues) == 0,
        "train_shape": list(X_train.shape),
        "test_shape": list(X_test.shape),
        "train_target_distribution": {str(k): int(v) for k, v in y_train.value_counts().items()},
        "test_target_distribution": {str(k): int(v) for k, v in y_test.value_counts().items()},
        "feature_count": int(X_train.shape[1]),
        "issues": issues,
    }


def save_validation_report(report: Dict[str, Any], output_path: str) -> str:
    """Write a validation summary to disk."""
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=4)
    return output_path
