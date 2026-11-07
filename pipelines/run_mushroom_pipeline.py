import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.mushroom_classification.pipeline import run_pipeline


if __name__ == "__main__":
    result = run_pipeline()
    print(f"Selected model: {result['selected_model']}")
    print(f"Metrics: {result['metrics']}")
    print(f"Model saved to: {result['model_path']}")
    if not result["model_replaced"]:
        print("The candidate did not replace the incumbent model.")
    if result["model_backup_path"]:
        print(f"Previous model backed up to: {result['model_backup_path']}")
