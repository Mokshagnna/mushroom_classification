import json
import os
from pathlib import Path

import mlflow
import yaml

from src.model_registry import MushroomModelRegistry

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    with open("config.yaml", "r", encoding="utf-8") as file:
        config = yaml.safe_load(file)
    tracking_uri = os.getenv("MLFLOW_TRACKING_URI") or config["mlflow"].get("tracking_uri")
    if tracking_uri:
        mlflow.set_tracking_uri(tracking_uri)

    model_name = config["mlflow"]["registry_model_name"]
    registry = MushroomModelRegistry(
        tracking_uri=tracking_uri,
        experiment_name=config["mlflow"]["experiment_name"],
    )
    matching_models = [
        model
        for model in registry.get_registry_summary()["registered_models"]
        if model["name"] == model_name
    ]
    if not matching_models:
        raise LookupError(f"Registered model not found: {model_name}")
    report = {
        "model_name": model_name,
        "versions": matching_models[0]["versions"],
    }
    report_path = PROJECT_ROOT / "reports" / "registry_report.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with report_path.open("w", encoding="utf-8") as file:
        json.dump(report, file, indent=2)


if __name__ == "__main__":
    main()
