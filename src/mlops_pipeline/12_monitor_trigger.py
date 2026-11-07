from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src.mlops_pipeline.monitoring_ops import run_monitoring_command


if __name__ == "__main__":
    run_monitoring_command("monitor_trigger")
