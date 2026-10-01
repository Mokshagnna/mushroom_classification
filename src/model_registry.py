"""MLflow model registration and version tracking for Mushroom Classification."""

import json
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

import mlflow
import mlflow.sklearn
from mlflow.tracking import MlflowClient

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


class MushroomModelRegistry:
    """Simplified model registry wrapper for the mushroom experiments."""

    def __init__(self, tracking_uri: Optional[str] = None, experiment_name: str = "mushroom-classification"):
        os.environ.setdefault("MLFLOW_ALLOW_FILE_STORE", "true")
        artifact_dir = Path(PROJECT_ROOT, "artifacts", "mlflow_artifacts")
        artifact_dir.mkdir(parents=True, exist_ok=True)

        db_file = Path(PROJECT_ROOT, "artifacts", "mlflow.db")
        db_file.parent.mkdir(parents=True, exist_ok=True)
        self.tracking_uri = f"sqlite:///{db_file.as_posix()}"
        self.artifact_uri = artifact_dir.as_uri()

        mlflow.set_tracking_uri(self.tracking_uri)
        mlflow.set_experiment(experiment_name)
        self.client = MlflowClient(tracking_uri=self.tracking_uri)
        self.experiment_name = experiment_name

    def _save_json(self, data: Dict[str, Any], path: str) -> str:
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(data, fh, indent=4)
        return path

    def log_model_run(
        self,
        model_name: str,
        model,
        params: Dict[str, Any],
        metrics: Dict[str, float],
        dataset_metadata: Dict[str, Any],
        preprocessing_metadata: Dict[str, Any],
        validation_metadata: Dict[str, Any],
        artifact_file_paths: Optional[Dict[str, str]] = None,
        classification_report: Optional[str] = None,
        confusion_matrix: Optional[Any] = None,
        model_stage: str = "candidate",
    ):
        """Log a run, save metadata, and register a model version."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_name = f"{model_name.replace(' ', '_').lower()}_{timestamp}"

        with mlflow.start_run(run_name=run_name) as run:
            mlflow.log_params({str(k): str(v) for k, v in params.items()})
            mlflow.log_metrics({str(k): float(v) for k, v in metrics.items()})

            metadata_path = os.path.join(PROJECT_ROOT, "artifacts", "run_metadata", f"{run_name}_metadata.json")
            self._save_json(
                {
                    "model_name": model_name,
                    "timestamp": datetime.now().isoformat(),
                    "dataset_metadata": dataset_metadata,
                    "preprocessing_metadata": preprocessing_metadata,
                    "validation_metadata": validation_metadata,
                    "stage": model_stage,
                },
                metadata_path,
            )
            mlflow.log_artifact(metadata_path, artifact_path="metadata")

            if classification_report is not None:
                report_path = os.path.join(PROJECT_ROOT, "artifacts", "run_metadata", f"{run_name}_classification_report.txt")
                with open(report_path, "w", encoding="utf-8") as fh:
                    fh.write(classification_report)
                mlflow.log_artifact(report_path, artifact_path="evaluation")

            if confusion_matrix is not None:
                confusion_path = os.path.join(PROJECT_ROOT, "artifacts", "run_metadata", f"{run_name}_confusion_matrix.json")
                with open(confusion_path, "w", encoding="utf-8") as fh:
                    json.dump(confusion_matrix.tolist(), fh, indent=4)
                mlflow.log_artifact(confusion_path, artifact_path="evaluation")

            if artifact_file_paths:
                for key, artifact_path in artifact_file_paths.items():
                    if os.path.exists(artifact_path):
                        mlflow.log_artifact(artifact_path, artifact_path=f"artifacts/{key}")

            mlflow.sklearn.log_model(model, artifact_path="model", serialization_format="cloudpickle")
            run_id = run.info.run_id

        try:
            registered_model = mlflow.register_model(model_uri=f"runs:/{run_id}/model", name=model_name)
            self.client.set_registered_model_alias(model_name, alias=model_stage, version=registered_model.version)
            registered = {
                "name": registered_model.name,
                "version": registered_model.version,
                "stage": model_stage,
            }
        except Exception as exc:  # pragma: no cover - MLflow registry may be unavailable in some setups.
            registered = {"error": str(exc), "run_id": run_id}

        return {
            "run_id": run_id,
            "registered_model": registered,
        }

    def get_registry_summary(self) -> Dict[str, Any]:
        """Return registered model versions for traceability and reporting."""
        models = self.client.search_registered_models()
        summary = []
        for model in models:
            versions = []
            for version in self.client.search_model_versions(f"name='{model.name}'"):
                versions.append(
                    {
                        "version": version.version,
                        "status": version.status,
                        "current_stage": version.current_stage,
                        "run_id": version.run_id,
                        "source": version.source,
                        "creation_timestamp": version.creation_timestamp,
                    }
                )
            summary.append({"name": model.name, "versions": versions})
        return {"registered_models": summary}
