from __future__ import annotations

import argparse
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src.mushroom_classification.inference import run_inference
from src.mushroom_classification.config import resolve_project_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Run mushroom batch inference.")
    parser.add_argument("--input", type=Path, required=True, help="CSV with the 22 mushroom features.")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs/inference_predictions.csv"),
        help="Prediction CSV output path.",
    )
    args = parser.parse_args()
    output = run_inference(resolve_project_path(args.input), resolve_project_path(args.output))
    print(f"Predictions saved to {output}")


if __name__ == "__main__":
    main()
