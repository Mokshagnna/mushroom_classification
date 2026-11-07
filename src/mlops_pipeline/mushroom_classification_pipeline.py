from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src.mlops_pipeline.mushroom_ops import run_stage
from src.mushroom_classification.config import resolve_project_path


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the staged mushroom MLOps workflow.")
    parser.add_argument("--config", type=Path, default=None)
    args = parser.parse_args()
    started = datetime.now(timezone.utc).isoformat()
    results = []
    try:
        for stage in range(1, 9):
            print(f"[MUSHROOM MLOPS] Stage {stage:02d}", flush=True)
            result = run_stage(stage, args.config)
            results.append({"stage": stage, "status": "PASSED", "summary": result})
    except Exception as error:
        results.append(
            {"stage": stage, "status": "FAILED", "error_type": type(error).__name__, "message": str(error)}
        )
        exit_code = 1
    else:
        exit_code = 0
    log = {
        "project": "mushroom-classification",
        "status": "PASSED" if exit_code == 0 else "FAILED",
        "started_at_utc": started,
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        "stages": results,
        "production_promotion": "not performed",
    }
    log_path = resolve_project_path("logs/mushroom_classification_pipeline_run_log.json")
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.write_text(json.dumps(log, indent=2), encoding="utf-8")
    print(f"Pipeline status: {log['status']}; run log: {log_path}")
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
