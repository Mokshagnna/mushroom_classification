from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from src.mushroom_classification.config import load_project_config, resolve_project_path
from src.mushroom_classification.evaluation import calculate_metrics
from src.mushroom_classification.inference import predict_frame
from src.mushroom_classification.monitoring import PredictionMonitor
from src.mushroom_classification.pipeline import run_pipeline, write_json
from src.mushroom_classification.validation import (
    FEATURE_COLUMNS,
    TARGET_COLUMN,
    encode_target,
    normalize_features,
    validate_dataset_schema,
    validate_inference_features,
)


def _read_events(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    events: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            if not line.strip():
                continue
            try:
                event = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"Malformed monitoring event at {path}:{line_number}.") from error
            if not isinstance(event, dict):
                raise ValueError(f"Monitoring event at {path}:{line_number} is not an object.")
            events.append(event)
    return events


def api_performance(config: dict[str, Any]) -> dict[str, Any]:
    path = resolve_project_path(config["api"]["monitoring_log"])
    events = _read_events(path)
    successful = [event for event in events if event.get("status") == "success"]
    latencies = np.asarray([float(event["latency_ms"]) for event in successful])
    return {
        "event_count": len(events),
        "successful_predictions": len(successful),
        "failed_requests": sum(event.get("status") == "failure" for event in events),
        "latency_ms": {
            "mean": float(latencies.mean()) if latencies.size else None,
            "p50": float(np.percentile(latencies, 50)) if latencies.size else None,
            "p95": float(np.percentile(latencies, 95)) if latencies.size else None,
            "max": float(latencies.max()) if latencies.size else None,
        },
        "accuracy": None,
        "accuracy_note": "No accuracy is inferred from unlabeled prediction requests.",
    }


def categorical_drift(reference: pd.DataFrame, current: pd.DataFrame) -> dict[str, Any]:
    reference_issues = validate_inference_features(reference.loc[:, list(FEATURE_COLUMNS)])
    current_issues = validate_inference_features(current.loc[:, list(FEATURE_COLUMNS)])
    if reference_issues:
        raise ValueError("Invalid reference feature data: " + "; ".join(reference_issues))
    if current_issues:
        raise ValueError("Invalid current feature data: " + "; ".join(current_issues))
    features: dict[str, Any] = {}
    distances = []
    for column in FEATURE_COLUMNS:
        ref = reference[column].fillna("__missing__").astype(str).value_counts(normalize=True)
        cur = current[column].fillna("__missing__").astype(str).value_counts(normalize=True)
        categories = set(ref.index) | set(cur.index)
        distance = 0.5 * sum(abs(float(ref.get(value, 0)) - float(cur.get(value, 0))) for value in categories)
        features[column] = {"total_variation": float(distance)}
        distances.append(distance)
    return {
        "method": "per-feature total variation distance",
        "feature_scores": features,
        "mean_drift_score": float(np.mean(distances)),
        "max_feature_drift": float(np.max(distances)),
        "feature_count": len(distances),
        "ground_truth_required": False,
    }


def prediction_drift(baseline_path: Path, current_path: Path) -> dict[str, Any]:
    baseline = [event for event in _read_events(baseline_path) if event.get("status") == "success"]
    current = [event for event in _read_events(current_path) if event.get("status") == "success"]
    if not baseline or not current:
        raise ValueError("Prediction drift requires non-empty baseline and current prediction logs.")
    counts: dict[str, Any] = {}
    distances = []
    for key, label in ((0, "edible"), (1, "poisonous")):
        baseline_rate = sum(int(event["prediction"]) == key for event in baseline) / len(baseline)
        current_rate = sum(int(event["prediction"]) == key for event in current) / len(current)
        counts[label] = {"baseline_rate": baseline_rate, "current_rate": current_rate}
        distances.append(abs(baseline_rate - current_rate))
    probability_delta = float(
        np.mean([float(event["poisonous_probability"]) for event in current])
        - np.mean([float(event["poisonous_probability"]) for event in baseline])
    )
    return {
        "baseline_predictions": len(baseline),
        "current_predictions": len(current),
        "prediction_rates": counts,
        "mean_poisonous_probability_delta": probability_delta,
        "total_variation": float(sum(distances) / 2),
        "accuracy": None,
        "note": "Prediction drift is not ground-truth performance or concept drift.",
    }


def labeled_performance(config: dict[str, Any], dataset_path: Path) -> dict[str, Any]:
    if not dataset_path.is_file():
        raise FileNotFoundError(f"Labeled mushroom dataset not found: {dataset_path}")
    frame = pd.read_csv(dataset_path)
    validation = validate_dataset_schema(frame)
    if not validation["valid"]:
        raise ValueError("Labeled dataset failed mushroom validation: " + "; ".join(validation["issues"]))
    model_path = resolve_project_path(config["paths"]["model"])
    if not model_path.is_file():
        raise FileNotFoundError(f"Saved mushroom model not found: {model_path}")
    model = joblib.load(model_path)
    prediction = predict_frame(model, frame.loc[:, list(FEATURE_COLUMNS)])
    labels = encode_target(frame[TARGET_COLUMN]).to_numpy()
    if set(np.unique(labels)) != {0, 1}:
        raise ValueError("Performance monitoring requires both edible and poisonous ground-truth labels.")
    metrics = calculate_metrics(
        labels,
        np.where(prediction["predicted_class"].to_numpy() == "poisonous", 1, 0),
        prediction["poisonous_probability"].to_numpy(),
    )
    return {
        "dataset": str(dataset_path),
        "ground_truth_source": "validated class column in supplied mushroom data",
        "rows": len(frame),
        "metrics": metrics,
        "evaluated_at_utc": datetime.now(timezone.utc).isoformat(),
    }


def run_monitoring_command(command: str, argv: list[str] | None = None) -> dict[str, Any]:
    parser = argparse.ArgumentParser(description=f"Mushroom {command.replace('_', ' ')}.")
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--input", type=Path, default=None)
    parser.add_argument("--baseline", type=Path, default=None)
    parser.add_argument("--current", type=Path, default=None)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--rows", type=int, default=1000)
    args = parser.parse_args(argv)
    if args.rows < 1:
        parser.error("--rows must be positive.")
    config = load_project_config(args.config)
    reports = resolve_project_path("reports")
    reports.mkdir(parents=True, exist_ok=True)

    if command == "api_performance":
        report = api_performance(config)
        output = resolve_project_path("artifacts/api_performance_report.json")
    elif command == "prediction_monitoring":
        report = PredictionMonitor(resolve_project_path(config["api"]["monitoring_log"])).summary()
        output = resolve_project_path(config["paths"]["monitoring_report"])
    elif command == "data_drift":
        if args.input is None:
            parser.error("--input must name current feature data for data_drift.")
        reference = pd.read_csv(resolve_project_path(config["data"]["raw_path"]))
        current = pd.read_csv(resolve_project_path(args.input))
        report = categorical_drift(reference, current)
        report["threshold"] = float(config["monitoring"]["data_drift_threshold"])
        report["status"] = (
            "DRIFT" if report["mean_drift_score"] >= report["threshold"] else "NO_DRIFT"
        )
        output = resolve_project_path(config["paths"]["drift_report"])
    elif command == "prediction_drift":
        if args.baseline is None or args.current is None:
            parser.error("--baseline and --current logs are required.")
        report = prediction_drift(
            resolve_project_path(args.baseline), resolve_project_path(args.current)
        )
        report["threshold"] = float(config["monitoring"]["prediction_drift_threshold"])
        report["status"] = (
            "DRIFT" if report["total_variation"] >= report["threshold"] else "NO_DRIFT"
        )
        output = resolve_project_path("reports/prediction_drift_report.json")
    elif command == "performance":
        if args.input is None:
            parser.error("--input must be a labeled dataset for performance monitoring.")
        report = labeled_performance(config, resolve_project_path(args.input))
        output = resolve_project_path(config["paths"]["performance_report"])
    elif command == "model_comparison":
        evaluation_path = resolve_project_path(config["paths"]["evaluation_report"])
        evaluation = json.loads(evaluation_path.read_text(encoding="utf-8"))
        report = {
            "selection_metric": evaluation["model_selection"]["selection_metric"],
            "candidates": evaluation["model_selection"]["candidates"],
            "selected_model": evaluation["model_selection"]["selected_model"],
        }
        output = resolve_project_path("reports/model_comparison_report.json")
    elif command == "create_evolved":
        source = pd.read_csv(resolve_project_path(config["data"]["raw_path"]))
        rng = np.random.default_rng(args.seed)
        sample = pd.DataFrame(
            {
                column: rng.choice(source[column].dropna().to_numpy(), size=args.rows, replace=True)
                for column in FEATURE_COLUMNS
            }
        )
        destination = resolve_project_path("data/monitoring/drift/evolved_data.csv")
        destination.parent.mkdir(parents=True, exist_ok=True)
        sample.to_csv(destination, index=False)
        report = {
            "dataset": str(destination),
            "rows": len(sample),
            "seed": args.seed,
            "kind": "synthetic label-free feature-distribution control",
            "not_ground_truth_or_real_observations": True,
        }
        output = resolve_project_path("reports/drift_dataset_metadata.json")
    elif command == "create_concept_drift":
        source = pd.read_csv(resolve_project_path(config["data"]["raw_path"]))
        rng = np.random.default_rng(args.seed)
        sample = source.sample(n=args.rows, replace=args.rows > len(source), random_state=args.seed).reset_index(drop=True)
        labels = encode_target(sample[TARGET_COLUMN]).to_numpy()
        # This controlled label perturbation is explicitly synthetic, never production truth.
        flip_mask = rng.random(args.rows) < 0.2
        altered = np.where(flip_mask, 1 - labels, labels)
        sample[TARGET_COLUMN] = np.where(altered == 1, "p", "e")
        destination = resolve_project_path("data/monitoring/drift/evolved_labeled_data.csv")
        destination.parent.mkdir(parents=True, exist_ok=True)
        sample.to_csv(destination, index=False)
        shutil.copy2(destination, destination.with_name("post_drift_training_data.csv"))
        report = {
            "dataset": str(destination),
            "rows": len(sample),
            "seed": args.seed,
            "synthetic_label_flip_rate": float(flip_mask.mean()),
            "warning": "Controlled synthetic labels are not reliable ground truth and must not be used for production accuracy claims.",
        }
        output = resolve_project_path("reports/concept_drift_dataset_metadata.json")
    elif command == "create_updated":
        if args.input is None:
            parser.error("--input must be a validated labeled dataset.")
        source_path = resolve_project_path(args.input)
        frame = pd.read_csv(source_path)
        validation = validate_dataset_schema(frame)
        if not validation["valid"]:
            raise ValueError("Updated dataset failed schema validation: " + "; ".join(validation["issues"]))
        destination = resolve_project_path("data/monitoring/drift/post_drift_training_data.csv")
        destination.parent.mkdir(parents=True, exist_ok=True)
        if source_path.resolve() != destination.resolve():
            shutil.copy2(source_path, destination)
        new_batch_path = resolve_project_path("data/raw/new_mushroom.csv")
        new_batch_path.parent.mkdir(parents=True, exist_ok=True)
        if source_path.resolve() != new_batch_path.resolve():
            shutil.copy2(source_path, new_batch_path)
        pd.DataFrame(
            {
                "valid": [validation["valid"]],
                "rows": [validation["row_count"]],
                "issues": ["; ".join(validation["issues"])],
            }
        ).to_csv(resolve_project_path(config["paths"]["new_data_validation_csv"]), index=False)
        report = {
            "source": str(source_path),
            "destination": str(destination),
            "new_batch_path": str(new_batch_path),
            "rows": len(frame),
            "dataset_sha256": hashlib.sha256(destination.read_bytes()).hexdigest(),
            "synthetic": False,
            "validation": "passed",
        }
        output = resolve_project_path("reports/updated_dataset_report.json")
    elif command == "concept_drift":
        if args.baseline is None or args.current is None:
            parser.error("--baseline and --current labeled datasets are required.")
        baseline_path = resolve_project_path(args.baseline)
        current_path = resolve_project_path(args.current)
        baseline_df = pd.read_csv(baseline_path)
        current_df = pd.read_csv(current_path)
        for name, frame in (("baseline", baseline_df), ("current", current_df)):
            validation = validate_dataset_schema(frame)
            if not validation["valid"]:
                raise ValueError(f"{name} labeled data is invalid: " + "; ".join(validation["issues"]))
        class_scores: dict[str, float] = {}
        for label, name in ((0, "edible"), (1, "poisonous")):
            left = baseline_df.loc[encode_target(baseline_df[TARGET_COLUMN]) == label]
            right = current_df.loc[encode_target(current_df[TARGET_COLUMN]) == label]
            if left.empty or right.empty:
                raise ValueError(f"Both labeled datasets must contain {name} rows.")
            distances = []
            for column in FEATURE_COLUMNS:
                left_rate = left[column].astype(str).value_counts(normalize=True)
                right_rate = right[column].astype(str).value_counts(normalize=True)
                categories = set(left_rate.index) | set(right_rate.index)
                distances.append(
                    0.5
                    * sum(
                        abs(float(left_rate.get(value, 0)) - float(right_rate.get(value, 0)))
                        for value in categories
                    )
                )
            class_scores[name] = float(np.mean(distances))
        synthetic = "evolved_labeled" in current_path.name or "concept_drift_dataset_metadata" in str(current_path)
        report = {
            "method": "class-conditional categorical total variation",
            "baseline": str(baseline_path),
            "current": str(current_path),
            "conditional_feature_scores": class_scores,
            "mean_conditional_shift": float(np.mean(list(class_scores.values()))),
            "synthetic_labels": synthetic,
            "reliable_ground_truth": not synthetic,
            "interpretation": (
                "Controlled synthetic experiment only; not a claim about real concept drift."
                if synthetic
                else "Statistical shift in labeled feature distributions; requires domain review."
            ),
        }
        output = resolve_project_path("reports/concept_drift_report.json")
    elif command == "request_history":
        log_path = resolve_project_path(config["api"]["monitoring_log"])
        events = _read_events(log_path)
        report = {
            "event_count": len(events),
            "events": events[-100:],
            "feature_values_included": False,
        }
        output = resolve_project_path("reports/request_history.json")
    elif command == "performance_trend":
        history = sorted(
            {
                *reports.glob("performance_*.json"),
                resolve_project_path(config["paths"]["performance_report"]),
            },
            key=lambda path: path.name,
        )
        points = []
        for path in history:
            if not path.is_file():
                continue
            document = json.loads(path.read_text(encoding="utf-8"))
            metrics = document.get("metrics", {})
            points.append(
                {
                    "file": path.name,
                    "timestamp": document.get("evaluated_at_utc"),
                    "poisonous_recall": metrics.get("poisonous_recall"),
                }
            )
        report = {
            "points": points,
            "status": "available" if points else "no_labeled_performance_history",
            "accuracy_note": "Only explicitly labeled reports are included.",
        }
        output = resolve_project_path("reports/performance_trend_report.json")
    elif command == "monitor_trigger":
        drift_path = resolve_project_path(config["paths"]["drift_report"])
        drift = json.loads(drift_path.read_text(encoding="utf-8")) if drift_path.exists() else None
        performance_path = resolve_project_path(config["paths"]["performance_report"])
        performance = (
            json.loads(performance_path.read_text(encoding="utf-8"))
            if performance_path.exists()
            else None
        )
        recall = (
            performance.get("metrics", {}).get("poisonous_recall")
            if performance is not None
            else None
        )
        report = {
            "drift_detected": bool(drift and drift.get("status") == "DRIFT"),
            "labeled_performance_available": recall is not None,
            "poisonous_recall": recall,
            "retraining_recommended": bool(
                (drift and drift.get("status") == "DRIFT")
                or (
                    recall is not None
                    and recall < float(config["monitoring"]["performance_recall_threshold"])
                )
            ),
            "automatic_retraining_or_promotion": False,
        }
        output = resolve_project_path("reports/monitor_trigger_report.json")
    elif command == "retrain":
        report = safe_retrain(config, args.config)
        output = resolve_project_path("reports/safe_retraining_report.json")
    elif command == "registry":
        from src.model_registry import MushroomModelRegistry

        uri = config["mlflow"].get("tracking_uri")
        registry = MushroomModelRegistry(uri, config["mlflow"]["experiment_name"])
        report = registry.get_registry_summary()
        report["promotion_performed"] = False
        output = resolve_project_path("reports/registry_report.json")
    else:
        raise ValueError(f"Unknown monitoring command: {command}")

    write_json(output, report)
    print(f"Report saved to {output}")
    return report


def safe_retrain(config: dict[str, Any], config_path: Path | None) -> dict[str, Any]:
    model_path = resolve_project_path(config["paths"]["model"])
    if not model_path.is_file():
        raise FileNotFoundError(
            f"Cannot compare retraining candidate; existing model is missing: {model_path}"
        )
    raw_path = resolve_project_path(config["data"]["raw_path"])
    raw = pd.read_csv(raw_path)
    features = normalize_features(raw)
    labels = encode_target(raw[TARGET_COLUMN])
    x_train, x_test, y_train, y_test = train_test_split_like(config, features, labels)
    incumbent = joblib.load(model_path)
    incumbent_frame = predict_frame(incumbent, x_test)
    incumbent_labels = np.where(
        incumbent_frame["predicted_class"].to_numpy() == "poisonous", 1, 0
    )
    incumbent_recall = float(np.mean(incumbent_labels[y_test.to_numpy() == 1] == 1))

    with tempfile.TemporaryDirectory(prefix="mushroom-retrain-") as directory:
        temp = Path(directory)
        candidate_config = dict(config)
        candidate_config["paths"] = dict(config["paths"])
        candidate_config["paths"]["model"] = str(temp / "candidate.joblib")
        candidate_config["paths"]["data_validation_report"] = str(temp / "validation.json")
        candidate_config["paths"]["evaluation_report"] = str(temp / "evaluation.json")
        candidate_config["mlflow"] = dict(config["mlflow"])
        candidate_config["mlflow"]["enabled"] = False
        candidate_path = temp / "config.yaml"
        candidate_path.write_text(yaml.safe_dump(candidate_config), encoding="utf-8")
        result = run_pipeline(candidate_path)
        candidate_report = json.loads((temp / "evaluation.json").read_text(encoding="utf-8"))
        candidate_recall = float(
            candidate_report["evaluation"]["metrics"]["poisonous_recall"]
        )
        passed_quality_gate = bool(candidate_report["quality_gate"]["passed"])
        replaced = passed_quality_gate and candidate_recall >= incumbent_recall
        if replaced:
            from src.mushroom_classification.pipeline import save_model_with_backup

            save_model_with_backup(joblib.load(result["model_path"]), model_path)
        return {
            "status": "candidate_accepted" if replaced else "incumbent_preserved",
            "incumbent_poisonous_recall": incumbent_recall,
            "candidate_poisonous_recall": candidate_recall,
            "quality_gate_passed": passed_quality_gate,
            "model_replaced": replaced,
            "comparison_split": "same deterministic stratified holdout",
            "automatic_registry_promotion": False,
        }


def train_test_split_like(
    config: dict[str, Any], features: pd.DataFrame, labels: pd.Series
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    split = config["split"]
    from sklearn.model_selection import train_test_split

    return train_test_split(
        features,
        labels,
        test_size=float(split["test_size"]),
        random_state=int(split["random_state"]),
        stratify=labels,
    )
