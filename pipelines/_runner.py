from pathlib import Path
import subprocess
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PIPELINE_DIR = PROJECT_ROOT / "src" / "mlops_pipeline"


def run_stages(stages: list[str]) -> int:
    for stage in stages:
        print(f"\n[PIPELINE] Running {stage}", flush=True)
        result = subprocess.run(
            [sys.executable, str(PIPELINE_DIR / stage)],
            cwd=PROJECT_ROOT,
        )
        if result.returncode:
            return result.returncode
    return 0
