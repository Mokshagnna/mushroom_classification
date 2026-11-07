import hashlib
import json
import subprocess
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROCESSED = PROJECT_ROOT / "data" / "processed"
FILES = ["X_train_final.npy", "X_test_final.npy", "y_train.npy", "y_test.npy"]


def hashes() -> dict[str, str]:
    return {
        name: hashlib.sha256((PROCESSED / name).read_bytes()).hexdigest()
        for name in FILES
    }


def main() -> int:
    stage = PROJECT_ROOT / "src" / "mlops_pipeline" / "03_preprocessing.py"
    first = subprocess.run([sys.executable, str(stage)], cwd=PROJECT_ROOT)
    if first.returncode:
        return first.returncode
    first_hashes = hashes()
    second = subprocess.run([sys.executable, str(stage)], cwd=PROJECT_ROOT)
    if second.returncode:
        return second.returncode
    second_hashes = hashes()
    report = {
        "status": "PASSED" if first_hashes == second_hashes else "FAILED",
        "files": {
            name: {
                "first_sha256": first_hashes[name],
                "second_sha256": second_hashes[name],
            }
            for name in FILES
        },
    }
    artifacts = PROJECT_ROOT / "artifacts"
    artifacts.mkdir(parents=True, exist_ok=True)
    serialized = json.dumps(report, indent=2)
    for filename in (
        "reproducibility_report.json",
        "pipeline_reproducibility_report.json",
    ):
        (artifacts / filename).write_text(serialized, encoding="utf-8")
    print(f"[{report['status']}] Preprocessing reproducibility check.")
    return 0 if report["status"] == "PASSED" else 1


if __name__ == "__main__":
    raise SystemExit(main())
