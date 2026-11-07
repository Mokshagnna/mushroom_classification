from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.mushroom_classification.config import load_project_config, resolve_project_path
from src.mushroom_classification.pipeline import write_json


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Validate prepared mushroom deployment; apply only with explicit approval."
    )
    parser.add_argument("--apply", action="store_true", help="Apply manifests with kubectl.")
    args = parser.parse_args()
    config = load_project_config()
    gate_path = resolve_project_path(config["paths"]["quality_gate_report"])
    metadata_path = resolve_project_path(config["paths"]["model_metadata"])
    for path in (gate_path, metadata_path):
        if not path.is_file():
            raise FileNotFoundError(f"Deployment prerequisite missing: {path}")
    gate = json.loads(gate_path.read_text(encoding="utf-8"))
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if gate.get("status") != "PASSED":
        raise RuntimeError("Deployment blocked because the model quality gate did not pass.")
    if metadata.get("deployment_status") != "prepared_not_deployed":
        raise RuntimeError("Deployment metadata is missing or does not identify a prepared model.")
    report = {
        "deployment_requested": args.apply,
        "status": "validated",
        "quality_gate": gate["status"],
        "model_sha256": metadata["model_sha256"],
        "note": "The project is an academic demonstration, not a food-safety service.",
    }
    if args.apply:
        executable = shutil.which("kubectl")
        if executable is None:
            raise FileNotFoundError("kubectl is required when --apply is supplied.")
        manifests = [
            resolve_project_path("k8s/deployment.yaml"),
            resolve_project_path("k8s/service.yaml"),
        ]
        result = subprocess.run(
            [executable, "apply", "-f", *(str(path) for path in manifests)],
            check=False,
            capture_output=True,
            text=True,
        )
        if result.returncode:
            raise RuntimeError(f"kubectl apply failed: {result.stderr.strip()}")
        report["status"] = "applied"
        report["kubectl_output"] = result.stdout.strip()
    destination = resolve_project_path("reports/deployment_automation_report.json")
    write_json(destination, report)
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
