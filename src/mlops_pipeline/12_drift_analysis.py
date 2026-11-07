from __future__ import annotations

import json
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src.mushroom_classification.config import resolve_project_path
from src.mushroom_classification.pipeline import write_json


def main() -> None:
    report_paths = {
        "data_drift": resolve_project_path("reports/data_drift_report.json"),
        "prediction_drift": resolve_project_path("reports/prediction_drift_report.json"),
        "concept_drift": resolve_project_path("reports/concept_drift_report.json"),
    }
    results = {
        name: json.loads(path.read_text(encoding="utf-8"))
        for name, path in report_paths.items()
        if path.is_file()
    }
    report = {
        "available_analyses": sorted(results),
        "results": results,
        "accuracy_claimed": False,
    }
    destination = resolve_project_path("reports/drift_analysis_report.json")
    write_json(destination, report)
    print(f"Drift analysis saved to {destination}")


if __name__ == "__main__":
    main()
