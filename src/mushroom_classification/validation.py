from __future__ import annotations

from typing import Any

import pandas as pd

TARGET_COLUMN = "class"
TARGET_LABELS = {"e": "edible", "p": "poisonous"}
FEATURE_COLUMNS = (
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
)
EXPECTED_COLUMNS = (TARGET_COLUMN, *FEATURE_COLUMNS)
FEATURE_CATEGORIES = {
    "cap-shape": {"b", "c", "f", "k", "s", "x"},
    "cap-surface": {"f", "g", "s", "y"},
    "cap-color": {"b", "c", "e", "g", "n", "p", "r", "u", "w", "y"},
    "bruises": {"f", "t"},
    "odor": {"a", "c", "f", "l", "m", "n", "p", "s", "y"},
    "gill-attachment": {"a", "f"},
    "gill-spacing": {"c", "w"},
    "gill-size": {"b", "n"},
    "gill-color": {"b", "e", "g", "h", "k", "n", "o", "p", "r", "u", "w", "y"},
    "stalk-shape": {"e", "t"},
    "stalk-root": {"?", "b", "c", "e", "r"},
    "stalk-surface-above-ring": {"f", "k", "s", "y"},
    "stalk-surface-below-ring": {"f", "k", "s", "y"},
    "stalk-color-above-ring": {"b", "c", "e", "g", "n", "o", "p", "w", "y"},
    "stalk-color-below-ring": {"b", "c", "e", "g", "n", "o", "p", "w", "y"},
    "veil-type": {"p"},
    "veil-color": {"n", "o", "w", "y"},
    "ring-number": {"n", "o", "t"},
    "ring-type": {"e", "f", "l", "n", "p"},
    "spore-print-color": {"b", "h", "k", "n", "o", "r", "u", "w", "y"},
    "population": {"a", "c", "n", "s", "v", "y"},
    "habitat": {"d", "g", "l", "m", "p", "u", "w"},
}


def validate_dataset_schema(df: pd.DataFrame) -> dict[str, Any]:
    issues: list[str] = []
    missing_columns = [column for column in EXPECTED_COLUMNS if column not in df.columns]
    extra_columns = [str(column) for column in df.columns if column not in EXPECTED_COLUMNS]

    if df.empty:
        issues.append("Dataset is empty.")
    if missing_columns:
        issues.append(f"Missing expected columns: {missing_columns}")
    if extra_columns:
        issues.append(f"Unexpected columns detected: {extra_columns}")
    invalid_categories: dict[str, list[str]] = {}
    for column in FEATURE_COLUMNS:
        if column not in df.columns:
            continue
        values = df[column].dropna().astype("string").str.strip().str.lower()
        invalid = sorted(set(values) - FEATURE_CATEGORIES[column])
        if invalid:
            invalid_categories[column] = invalid
            issues.append(f"Unsupported categories in column '{column}': {invalid}")

    missing_values: dict[str, int] = {}
    for column in df.columns:
        null_count = int(df[column].isna().sum())
        blank_count = int(
            df[column].map(lambda value: isinstance(value, str) and not value.strip()).sum()
        )
        if null_count + blank_count:
            missing_values[str(column)] = null_count + blank_count
            issues.append(f"Missing or blank values detected in column '{column}'.")

    invalid_targets: list[str] = []
    class_distribution: dict[str, int] = {}
    if TARGET_COLUMN not in df.columns:
        issues.append(f"Target column '{TARGET_COLUMN}' is missing.")
    else:
        targets = df[TARGET_COLUMN].astype("string").str.strip().str.lower()
        invalid_targets = sorted(
            str(value) for value in targets.dropna().unique() if value not in TARGET_LABELS
        )
        if invalid_targets:
            issues.append(f"Unsupported target values: {invalid_targets}")
        class_distribution = {
            str(value): int(count)
            for value, count in targets.value_counts(dropna=False).items()
        }

    duplicate_rows = int(df.duplicated().sum())
    if duplicate_rows:
        issues.append(f"Duplicate rows detected: {duplicate_rows}")

    constant_features = [
        column
        for column in FEATURE_COLUMNS
        if column in df.columns and df[column].nunique(dropna=False) <= 1
    ]
    unknown_stalk_root_values = 0
    if "stalk-root" in df.columns:
        unknown_stalk_root_values = int(
            df["stalk-root"].astype("string").str.strip().eq("?").sum()
        )

    return {
        "valid": not issues,
        "expected_columns": list(EXPECTED_COLUMNS),
        "missing_columns": missing_columns,
        "extra_columns": extra_columns,
        "row_count": int(df.shape[0]),
        "column_count": int(df.shape[1]),
        "missing_values": missing_values,
        "unknown_stalk_root_values": unknown_stalk_root_values,
        "duplicate_rows": duplicate_rows,
        "constant_features": constant_features,
        "class_distribution": class_distribution,
        "invalid_targets": invalid_targets,
        "invalid_categories": invalid_categories,
        "issues": issues,
    }


def validate_inference_features(df: pd.DataFrame) -> list[str]:
    allowed_columns = set(FEATURE_COLUMNS) | {TARGET_COLUMN}
    missing = [column for column in FEATURE_COLUMNS if column not in df.columns]
    extra = [str(column) for column in df.columns if column not in allowed_columns]
    issues = []
    if df.empty:
        issues.append("Inference data is empty.")
    if missing:
        issues.append(f"Missing expected feature columns: {missing}")
    if extra:
        issues.append(f"Unexpected columns detected: {extra}")
    return issues


def normalize_features(df: pd.DataFrame) -> pd.DataFrame:
    features = df.loc[:, list(FEATURE_COLUMNS)].copy()
    features["stalk-root"] = features["stalk-root"].replace("?", pd.NA)
    return features


def encode_target(target: pd.Series) -> pd.Series:
    normalized = target.astype("string").str.strip().str.lower()
    encoded = normalized.map({"e": 0, "p": 1})
    if encoded.isna().any():
        invalid = sorted(str(value) for value in normalized[encoded.isna()].unique())
        raise ValueError(f"Unsupported target values: {invalid}")
    encoded.name = TARGET_COLUMN
    return encoded.astype("int64")
