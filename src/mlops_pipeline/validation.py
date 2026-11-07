from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.mushroom_classification.config import resolve_project_path
from src.mushroom_classification.validation import (
    EXPECTED_COLUMNS,
    FEATURE_CATEGORIES,
    TARGET_COLUMN,
    TARGET_LABELS,
    validate_dataset_schema,
)

CATEGORY_VALUES = dict(FEATURE_CATEGORIES)
CATEGORY_VALUES[TARGET_COLUMN] = set(TARGET_LABELS)


def validate_schema(
    frame: pd.DataFrame, output_report_name: str | None = None
) -> bool:
    report = validate_dataset_schema(frame)
    if output_report_name:
        destination = resolve_project_path(Path("artifacts") / output_report_name)
        destination.parent.mkdir(parents=True, exist_ok=True)
        issues = report["issues"]
        pd.DataFrame({"issue": issues}, columns=["issue"]).to_csv(
            destination, index=False
        )
    return bool(report["valid"])


__all__ = ["EXPECTED_COLUMNS", "CATEGORY_VALUES", "validate_schema"]
