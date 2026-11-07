from __future__ import annotations

import json
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src.mushroom_classification.config import load_project_config, resolve_project_path
from src.mushroom_classification.pipeline import write_json


def main() -> int:
    config = load_project_config()
    metadata_path = resolve_project_path(config["paths"]["model_metadata"])
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    gate_path = resolve_project_path(config["paths"]["quality_gate_report"])
    gate = json.loads(gate_path.read_text(encoding="utf-8"))
    eligible = gate.get("status") == "PASSED" and metadata.get("deployment_status") == "prepared_not_deployed"
    report = {
        "deployment_performed": False,
        "reason": (
            "No explicit production deployment approval is configured."
            if eligible
            else "Model deployment preparation or quality gate is incomplete."
        ),
        "quality_gate": gate.get("status"),
    }
    write_json(resolve_project_path("reports/deployment_automation_report.json"), report)
    print(report["reason"])
    return 0 if eligible else 1


if __name__ == "__main__":
    raise SystemExit(main())
