from __future__ import annotations

import hashlib
import argparse
import json
import math
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from src.mushroom_classification.config import (
    load_project_config,
    resolve_project_path,
)
from src.mushroom_classification.inference import predict_frame
from src.mushroom_classification.models import build_preprocessor
from src.mushroom_classification.pipeline import run_pipeline, write_json
from src.mushroom_classification.validation import (
    FEATURE_COLUMNS,
    TARGET_COLUMN,
    encode_target,
    normalize_features,
    validate_dataset_schema,
)


def configured_path(config: dict[str, Any], key: str) -> Path:
    try:
        return resolve_project_path(config["paths"][key])
    except (KeyError, TypeError) as error:
        raise ValueError(f"Configuration must define paths.{key}.") from error


def load_dataset(path: Path) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(f"Mushroom dataset not found: {path}")
    try:
        return pd.read_csv(path)
    except (pd.errors.ParserError, UnicodeDecodeError) as error:
        raise ValueError(f"Could not parse mushroom dataset: {path}") from error


def _jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    raise TypeError(f"Cannot serialize {type(value).__name__}.")


def run_stage_cli(stage: int) -> None:
    parser = argparse.ArgumentParser(description=f"Run mushroom MLOps stage {stage:02d}.")
    parser.add_argument("--config", type=Path, default=None)
    args = parser.parse_args()
    print(json.dumps(run_stage(stage, args.config), indent=2, default=_jsonable))


def run_stage(stage: int, config_path: Path | None = None) -> dict[str, Any]:
    config = load_project_config(config_path)
    raw_path = resolve_project_path(config["data"]["raw_path"])
    processed_dir = resolve_project_path(config["data"]["processed_dir"])
    raw_df: pd.DataFrame | None = None

    if stage in (1, 2, 3, 5, 6, 9):
        raw_df = load_dataset(raw_path)

    if stage == 1:
        report = {
            "stage": "mushroom_data_loading",
            "loaded_at_utc": datetime.now(timezone.utc).isoformat(),
            "source_path": str(raw_path),
            "sha256": hashlib.sha256(raw_path.read_bytes()).hexdigest(),
            "row_count": len(raw_df),
            "columns": list(raw_df.columns),
            "target_column": TARGET_COLUMN,
            "class_mapping": {"e": 0, "p": 1},
        }
        write_json(configured_path(config, "data_load_report"), report)
        return report

    if stage == 2:
        validation = validate_dataset_schema(raw_df)
        validation["source_path"] = str(raw_path)
        write_json(configured_path(config, "data_validation_report"), validation)
        category_rows = [
            {
                "feature": column,
                "observed_categories": ",".join(
                    sorted(raw_df[column].dropna().astype(str).unique())
                ) if column in raw_df.columns else "",
                "valid": (
                    column in raw_df.columns
                    and not validation.get("invalid_categories", {}).get(column)
                ),
            }
            for column in FEATURE_COLUMNS
        ]
        baseline_csv = configured_path(config, "baseline_validation_csv")
        baseline_csv.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(category_rows).to_csv(baseline_csv, index=False)
        if not validation["valid"]:
            raise ValueError("Mushroom dataset validation failed: " + "; ".join(validation["issues"]))
        return validation

    if stage == 3:
        validation = validate_dataset_schema(raw_df)
        if not validation["valid"]:
            raise ValueError("Cannot preprocess invalid data: " + "; ".join(validation["issues"]))
        features = normalize_features(raw_df)
        labels = encode_target(raw_df[TARGET_COLUMN])
        split = config["split"]
        x_train, x_test, y_train, y_test = train_test_split(
            features,
            labels,
            test_size=float(split["test_size"]),
            random_state=int(split["random_state"]),
            stratify=labels,
        )
        preprocessor = build_preprocessor()
        train_matrix = preprocessor.fit_transform(x_train)
        test_matrix = preprocessor.transform(x_test)
        if hasattr(train_matrix, "toarray"):
            train_matrix = train_matrix.toarray()
        if hasattr(test_matrix, "toarray"):
            test_matrix = test_matrix.toarray()
        processed_dir.mkdir(parents=True, exist_ok=True)
        np.save(processed_dir / "X_train_final.npy", np.asarray(train_matrix))
        np.save(processed_dir / "X_test_final.npy", np.asarray(test_matrix))
        np.save(processed_dir / "y_train.npy", y_train.to_numpy(dtype=np.int64))
        np.save(processed_dir / "y_test.npy", y_test.to_numpy(dtype=np.int64))
        x_train.to_csv(processed_dir / "X_train_raw.csv", index=False)
        x_test.to_csv(processed_dir / "X_test_raw.csv", index=False)
        joblib.dump(preprocessor, configured_path(config, "preprocessor"))
        metadata = {
            "dataset": str(raw_path),
            "target_column": TARGET_COLUMN,
            "target_mapping": {"0": "edible", "1": "poisonous"},
            "feature_columns": list(FEATURE_COLUMNS),
            "train_shape": list(np.asarray(train_matrix).shape),
            "test_shape": list(np.asarray(test_matrix).shape),
            "split_random_state": int(split["random_state"]),
            "unknown_stalk_root_marker": "?",
            "preprocessing": "training-only imputation and one-hot encoding",
        }
        write_json(processed_dir / "dataset_metadata.json", metadata)
        write_json(configured_path(config, "preprocessing_report"), metadata)
        return metadata

    if stage == 4:
        paths = {
            "x_train": processed_dir / "X_train_final.npy",
            "x_test": processed_dir / "X_test_final.npy",
            "y_train": processed_dir / "y_train.npy",
            "y_test": processed_dir / "y_test.npy",
            "x_train_raw": processed_dir / "X_train_raw.csv",
            "x_test_raw": processed_dir / "X_test_raw.csv",
        }
        missing = [str(path) for path in paths.values() if not path.is_file()]
        if missing:
            raise FileNotFoundError(
                "Processed data is incomplete; run stage 03 first: " + ", ".join(missing)
            )
        x_train = np.load(paths["x_train"], allow_pickle=False)
        x_test = np.load(paths["x_test"], allow_pickle=False)
        y_train = np.load(paths["y_train"], allow_pickle=False)
        y_test = np.load(paths["y_test"], allow_pickle=False)
        errors: list[str] = []
        if x_train.ndim != 2 or x_test.ndim != 2:
            errors.append("Transformed features must be 2-dimensional.")
        if x_train.shape[1] != x_test.shape[1]:
            errors.append("Train and test transformed feature counts differ.")
        if len(y_train) != len(x_train) or len(y_test) != len(x_test):
            errors.append("Feature and target row counts differ.")
        if not np.isfinite(x_train).all() or not np.isfinite(x_test).all():
            errors.append("Transformed feature matrices contain non-finite values.")
        if set(np.unique(y_train)) != {0, 1} or set(np.unique(y_test)) != {0, 1}:
            errors.append("Train and test labels must both include 0 and 1.")
        report = {
            "status": "FAILED" if errors else "PASSED",
            "shapes": {
                name: list(np.load(path, mmap_mode="r", allow_pickle=False).shape)
                for name, path in paths.items()
                if path.suffix == ".npy"
            },
            "errors": errors,
        }
        write_json(configured_path(config, "preprocessing_validation_report"), report)
        if errors:
            raise ValueError("Processed output validation failed: " + "; ".join(errors))
        return report

    if stage == 5:
        result = run_pipeline(config_path)
        report = {
            "stage": "mushroom_training_and_registry",
            "selected_model": result["selected_model"],
            "model_path": result["model_path"],
            "model_replaced": result["model_replaced"],
            "incumbent_poisonous_recall": result["incumbent_poisonous_recall"],
            "evaluation_metrics": result["metrics"],
            "registry_result": result["registry_result"],
            "promotion": "not performed; registered versions remain candidates",
        }
        write_json(configured_path(config, "production_model_report"), report)
        return report

    if stage == 6:
        model_path = configured_path(config, "model")
        if not model_path.is_file():
            raise FileNotFoundError(f"Saved mushroom model not found: {model_path}")
        report_path = configured_path(config, "evaluation_report")
        report = json.loads(report_path.read_text(encoding="utf-8"))
        metrics = report.get("evaluation", {}).get("metrics")
        if not isinstance(metrics, dict) or not metrics.get("confusion_matrix"):
            raise ValueError(f"Evaluation metrics are missing from {report_path}")
        matrix = np.asarray(metrics["confusion_matrix"])
        if matrix.shape != (2, 2):
            raise ValueError("Evaluation confusion matrix must be 2x2.")
        confusion_path = configured_path(config, "confusion_matrix_plot")
        confusion_path.parent.mkdir(parents=True, exist_ok=True)
        from sklearn.metrics import ConfusionMatrixDisplay, RocCurveDisplay

        ConfusionMatrixDisplay(matrix, display_labels=["edible", "poisonous"]).plot()
        plt.tight_layout()
        plt.savefig(confusion_path)
        plt.close()
        x_test_path = processed_dir / "X_test_raw.csv"
        y_test_path = processed_dir / "y_test.npy"
        if x_test_path.is_file() and y_test_path.is_file():
            x_test = pd.read_csv(x_test_path)
            y_test = np.load(y_test_path, allow_pickle=False)
            if len(np.unique(y_test)) == 2:
                model = joblib.load(model_path)
                result = predict_frame(model, x_test)
                probabilities = result["poisonous_probability"].to_numpy()
                RocCurveDisplay.from_predictions(y_test, probabilities)
                plt.tight_layout()
                roc_path = configured_path(config, "roc_curve_plot")
                roc_path.parent.mkdir(parents=True, exist_ok=True)
                plt.savefig(roc_path)
                plt.close()
                report["roc_curve"] = "generated from labeled held-out test data"
            else:
                report["roc_curve"] = "not generated: test data lacks both labels"
        else:
            report["roc_curve"] = "not generated: labeled processed test split is unavailable"
        return report

    if stage == 7:
        report = json.loads(configured_path(config, "evaluation_report").read_text(encoding="utf-8"))
        metrics = report.get("evaluation", {}).get("metrics", {})
        metric_name = config["quality_gate"]["metric"]
        minimum = float(config["quality_gate"]["minimum_poisonous_recall"])
        if metric_name not in metrics:
            raise KeyError(f"Quality-gate metric is absent from evaluation report: {metric_name}")
        actual = float(metrics[metric_name])
        passed = math.isfinite(actual) and actual >= minimum
        gate = {
            "status": "PASSED" if passed else "FAILED",
            "metric": metric_name,
            "minimum_value": minimum,
            "actual_value": actual,
            "model_replacement_allowed": passed,
        }
        write_json(configured_path(config, "quality_gate_report"), gate)
        if not passed:
            raise RuntimeError(f"Model quality gate failed: {metric_name}={actual:.4f} < {minimum:.4f}")
        return gate

    if stage == 8:
        gate_path = configured_path(config, "quality_gate_report")
        if not gate_path.is_file():
            raise FileNotFoundError("Run stage 07 before lifecycle checks.")
        gate = json.loads(gate_path.read_text(encoding="utf-8"))
        report = {
            "status": "PASSED" if gate.get("status") == "PASSED" else "BLOCKED",
            "quality_gate": gate,
            "promotion_performed": False,
            "decision": (
                "Candidate is eligible for review; automatic production promotion is disabled."
                if gate.get("status") == "PASSED"
                else "Candidate is blocked by the quality gate."
            ),
            "recorded_at_utc": datetime.now(timezone.utc).isoformat(),
        }
        write_json(configured_path(config, "lifecycle_report"), report)
        if report["status"] != "PASSED":
            raise RuntimeError("Lifecycle is blocked because the quality gate did not pass.")
        return report

    if stage == 9:
        model_path = configured_path(config, "model")
        if not model_path.is_file():
            raise FileNotFoundError(f"Saved mushroom model not found: {model_path}")
        return {"model_path": str(model_path), "feature_count": len(FEATURE_COLUMNS)}

    if stage == 10:
        return run_failure_scenarios(config)

    if stage == 11:
        model_path = configured_path(config, "model")
        if not model_path.is_file():
            raise FileNotFoundError(f"Cannot prepare deployment; model missing: {model_path}")
        model = joblib.load(model_path)
        metadata = {
            "model_path": str(model_path),
            "model_sha256": hashlib.sha256(model_path.read_bytes()).hexdigest(),
            "model_type": type(model).__name__,
            "classes": [int(value) for value in model.classes_],
            "class_mapping": {"0": "edible", "1": "poisonous"},
            "feature_columns": list(FEATURE_COLUMNS),
            "deployment_status": "prepared_not_deployed",
            "safety_notice": "Academic demonstration only; not a food-safety tool.",
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
        }
        write_json(configured_path(config, "model_metadata"), metadata)
        return metadata

    raise ValueError(f"Unsupported MLOps stage: {stage}")


def run_failure_scenarios(config: dict[str, Any]) -> dict[str, Any]:
    from fastapi.testclient import TestClient

    from api.main import create_app, load_model
    raw_path = resolve_project_path(config["data"]["raw_path"])
    rows = []

    def expect_failure(name: str, action: Any) -> None:
        try:
            action()
        except Exception as error:
            rows.append({"scenario": name, "passed": True, "observed": type(error).__name__})
        else:
            rows.append({"scenario": name, "passed": False, "observed": "no error"})

    with tempfile.TemporaryDirectory(prefix="mushroom-failures-") as directory:
        temp = Path(directory)
        expect_failure("missing_dataset", lambda: load_dataset(temp / "missing.csv"))
        expect_failure(
            "invalid_feature_columns",
            lambda: _raise_if_invalid_schema(pd.DataFrame({"unexpected": ["x"]})),
        )
        invalid = pd.read_csv(raw_path).head(2)
        invalid.loc[invalid.index[0], FEATURE_COLUMNS[0]] = "__invalid__"
        expect_failure("invalid_categorical_value", lambda: _raise_if_invalid_schema(invalid))
        corrupt = temp / "corrupt.csv"
        corrupt.write_bytes(b"\x00\x00\x00")
        expect_failure("corrupted_input", lambda: _load_corrupt_csv(corrupt))
        expect_failure("missing_model", lambda: load_model(temp / "missing.joblib"))
        corrupt_model = temp / "corrupt.joblib"
        corrupt_model.write_text("not a model", encoding="utf-8")
        expect_failure("model_loading_failure", lambda: joblib.load(corrupt_model))

        model_path = configured_path(config, "model")
        from src.mushroom_classification.monitoring import PredictionMonitor

        app = create_app(
            model_path=model_path,
            monitor=PredictionMonitor(temp / "api-monitor.jsonl"),
        )
        features = pd.read_csv(raw_path).iloc[[0]].loc[:, list(FEATURE_COLUMNS)]
        payload = features.iloc[0].to_dict()
        payload[FEATURE_COLUMNS[0]] = "__invalid__"
        with TestClient(app) as client:
            health = client.get("/health")
            valid = client.post("/predict", json=features.iloc[0].to_dict())
            response = client.post("/predict", json=payload)
        api_report = {
            "health_status": health.status_code,
            "valid_prediction_status": valid.status_code,
            "invalid_request_status": response.status_code,
            "all_checks_passed": (
                health.status_code == 200
                and valid.status_code == 200
                and response.status_code == 422
            ),
        }
        write_json(configured_path(config, "api_test_report"), api_report)
        rows.append(
            {
                "scenario": "invalid_api_request",
                "passed": response.status_code == 422,
                "observed": response.status_code,
            }
        )

    report = {
        "status": "PASSED" if all(row["passed"] for row in rows) else "FAILED",
        "scenarios": rows,
        "real_dataset_or_model_modified": False,
    }
    write_json(configured_path(config, "failure_report"), report)
    write_json(configured_path(config, "failure_simulation_report"), report)
    if report["status"] != "PASSED":
        raise RuntimeError("At least one isolated failure-injection scenario did not behave as expected.")
    return report


def _raise_if_invalid_schema(frame: pd.DataFrame) -> None:
    result = validate_dataset_schema(frame)
    if not result["valid"]:
        raise ValueError("; ".join(result["issues"]))


def _load_corrupt_csv(path: Path) -> None:
    frame = pd.read_csv(path)
    if frame.empty:
        raise ValueError("Corrupt dataset is empty.")
