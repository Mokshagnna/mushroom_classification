from __future__ import annotations

import hashlib
import json
import os
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.metrics import recall_score

from .config import PROJECT_ROOT, load_project_config, resolve_project_path
from .evaluation import calculate_metrics
from .models import build_model_pipeline
from .validation import TARGET_COLUMN, encode_target, normalize_features, validate_dataset_schema

MODEL_CANDIDATES = (
    "Logistic Regression",
    "Decision Tree",
    "Random Forest",
    "K-Nearest Neighbors",
)


def load_config(config_path: Path | None = None) -> dict[str, Any]:
    return load_project_config(config_path)


def resolve_path(path: str) -> Path:
    return resolve_project_path(path)


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def save_model_with_backup(estimator: Any, model_path: Path) -> Path | None:
    model_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = model_path.with_name(f".{model_path.name}.tmp")
    backup_path = None
    try:
        joblib.dump(estimator, temporary_path)
        if model_path.exists():
            timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
            backup_path = model_path.with_name(
                f"{model_path.stem}_{timestamp}.bak{model_path.suffix}"
            )
            shutil.copy2(model_path, backup_path)
        os.replace(temporary_path, model_path)
    finally:
        if temporary_path.exists():
            temporary_path.unlink()
    return backup_path


def compare_models(
    x_train: pd.DataFrame,
    y_train: pd.Series,
    random_state: int,
    n_estimators: int,
    cv_folds: int,
) -> list[dict[str, Any]]:
    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=random_state)
    scores = []
    for name in MODEL_CANDIDATES:
        estimator = build_model_pipeline(name, random_state, n_estimators)
        fold_scores = cross_val_score(
            estimator,
            x_train,
            y_train,
            cv=cv,
            scoring="recall",
            n_jobs=1,
        )
        scores.append(
            {
                "model": name,
                "poisonous_recall_mean": float(fold_scores.mean()),
                "poisonous_recall_std": float(fold_scores.std()),
            }
        )
    return sorted(
        scores,
        key=lambda result: (result["poisonous_recall_mean"], -result["poisonous_recall_std"]),
        reverse=True,
    )


def _poisonous_probabilities(estimator: Any, x_test: pd.DataFrame) -> np.ndarray:
    classes = list(estimator.classes_)
    if 1 not in classes:
        raise ValueError("The fitted model does not contain the poisonous class.")
    return estimator.predict_proba(x_test)[:, classes.index(1)]


def run_pipeline(config_path: Path | None = None) -> dict[str, Any]:
    config = load_config(config_path)
    raw_path = resolve_path(config["data"]["raw_path"])
    validation_report_path = resolve_path(config["paths"]["data_validation_report"])
    evaluation_report_path = resolve_path(config["paths"]["evaluation_report"])
    model_path = resolve_path(config["paths"]["model"])

    if not raw_path.is_file():
        raise FileNotFoundError(f"Mushroom dataset not found: {raw_path}")

    raw_df = pd.read_csv(raw_path)
    validation = validate_dataset_schema(raw_df)
    validation["dataset_sha256"] = hashlib.sha256(raw_path.read_bytes()).hexdigest()
    write_json(validation_report_path, validation)
    if not validation["valid"]:
        raise ValueError(
            f"Dataset validation failed; see {validation_report_path}: "
            + "; ".join(validation["issues"])
        )

    x = normalize_features(raw_df)
    y = encode_target(raw_df[TARGET_COLUMN])
    split_config = config["split"]
    x_train, x_test, y_train, y_test = train_test_split(
        x,
        y,
        test_size=float(split_config["test_size"]),
        random_state=int(split_config["random_state"]),
        stratify=y,
    )

    model_config = config["model"]
    random_state = int(model_config["random_state"])
    n_estimators = int(model_config["n_estimators"])
    cv_folds = int(model_config["cv_folds"])
    comparisons = compare_models(x_train, y_train, random_state, n_estimators, cv_folds)
    selected_model = comparisons[0]["model"]
    estimator = build_model_pipeline(selected_model, random_state, n_estimators)
    estimator.fit(x_train, y_train)

    y_pred = estimator.predict(x_test)
    metrics = calculate_metrics(
        y_test.to_numpy(),
        np.asarray(y_pred),
        _poisonous_probabilities(estimator, x_test),
    )
    gate_config = config["quality_gate"]
    gate_metric = str(gate_config["metric"])
    if gate_metric != "poisonous_recall":
        raise ValueError(
            "The supported quality-gate metric is 'poisonous_recall'; "
            f"received {gate_metric!r}."
        )
    minimum_recall = float(gate_config["minimum_poisonous_recall"])
    gate_passed = metrics[gate_metric] >= minimum_recall
    incumbent_recall = None
    model_replaced = True
    if model_path.is_file():
        incumbent = joblib.load(model_path)
        if not callable(getattr(incumbent, "predict", None)):
            raise ValueError(f"Existing model artifact cannot predict: {model_path}")
        incumbent_predictions = np.asarray(incumbent.predict(x_test), dtype=int)
        incumbent_recall = float(
            recall_score(y_test, incumbent_predictions, pos_label=1, zero_division=0)
        )
        model_replaced = metrics["poisonous_recall"] >= incumbent_recall
    report = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "dataset": {
            "path": str(raw_path),
            "rows": int(len(raw_df)),
            "sha256": validation["dataset_sha256"],
            "target_column": TARGET_COLUMN,
            "target_mapping": {"0": "edible", "1": "poisonous"},
            "test_size": float(split_config["test_size"]),
            "random_state": int(split_config["random_state"]),
        },
        "model_selection": {
            "selection_metric": "poisonous_recall",
            "cross_validation_folds": cv_folds,
            "candidates": comparisons,
            "selected_model": selected_model,
        },
        "evaluation": {
            "split": "held-out stratified test set",
            "rows": int(len(y_test)),
            "metrics": metrics,
        },
        "quality_gate": {
            "metric": gate_metric,
            "minimum_value": minimum_recall,
            "actual_value": metrics["poisonous_recall"],
            "passed": gate_passed,
        },
        "incumbent_comparison": {
            "incumbent_poisonous_recall": incumbent_recall,
            "candidate_poisonous_recall": metrics["poisonous_recall"],
            "candidate_replaces_incumbent": model_replaced if incumbent_recall is not None else True,
            "comparison_split": "same deterministic stratified held-out test split",
        },
        "safety_notice": (
            "This model is an academic demonstration, not a food-safety tool. "
            "Never use its predictions to decide whether a mushroom is safe to eat."
        ),
    }
    write_json(evaluation_report_path, report)
    if not gate_passed:
        raise RuntimeError(
            "Poisonous-recall quality gate failed: "
            f"{metrics['poisonous_recall']:.4f} < {minimum_recall:.4f}. "
            f"See {evaluation_report_path}."
        )

    model_backup_path = (
        save_model_with_backup(estimator, model_path) if model_replaced else None
    )

    registry_result = None
    mlflow_config = config["mlflow"]
    if mlflow_config.get("enabled", False):
        from src.model_registry import MushroomModelRegistry

        tracking_uri = os.getenv("MLFLOW_TRACKING_URI") or mlflow_config["tracking_uri"]
        registry = MushroomModelRegistry(
            tracking_uri=tracking_uri,
            experiment_name=mlflow_config["experiment_name"],
        )
        registry_result = registry.log_model_run(
            model_name=mlflow_config["registry_model_name"],
            model=estimator,
            params={
                "selected_model": selected_model,
                "random_state": random_state,
                "n_estimators": n_estimators,
                "cv_folds": cv_folds,
            },
            metrics={
                key: value
                for key, value in metrics.items()
                if key != "confusion_matrix"
                and key != "confusion_matrix_labels"
                and key != "classification_report"
            },
            dataset_metadata=validation,
            preprocessing_metadata={
                "strategy": "training-fold categorical imputation and one-hot encoding",
                "unknown_category_handling": "ignore unseen categories",
                "missing_stalk_root_marker": "?",
            },
            validation_metadata=validation,
            artifact_file_paths={
                "evaluation_report": str(evaluation_report_path),
                "data_validation_report": str(validation_report_path),
            },
            classification_report=json.dumps(metrics["classification_report"], indent=2),
            confusion_matrix=np.asarray(metrics["confusion_matrix"]),
            model_stage="candidate",
        )

    return {
        "model_path": str(model_path),
        "model_backup_path": str(model_backup_path) if model_backup_path else None,
        "model_replaced": model_replaced,
        "incumbent_poisonous_recall": incumbent_recall,
        "validation_report_path": str(validation_report_path),
        "evaluation_report_path": str(evaluation_report_path),
        "selected_model": selected_model,
        "metrics": metrics,
        "registry_result": registry_result,
    }
