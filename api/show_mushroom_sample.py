from __future__ import annotations

import json
from pathlib import Path
import sys

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.mushroom_classification.config import load_project_config, resolve_project_path
from src.mushroom_classification.validation import FEATURE_COLUMNS


def main() -> None:
    config = load_project_config()
    dataset_path = resolve_project_path(config["data"]["raw_path"])
    if not dataset_path.is_file():
        raise FileNotFoundError(f"Mushroom dataset not found: {dataset_path}")
    sample = pd.read_csv(dataset_path, usecols=list(FEATURE_COLUMNS)).iloc[0].to_dict()
    print(json.dumps(sample, indent=2))
    print(f"Sample source: {dataset_path}")
    print("This is an existing dataset row, not a generated or synthetic example.")


if __name__ == "__main__":
    main()
