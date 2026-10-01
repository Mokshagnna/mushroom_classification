"""Lab 4 MLflow tracking pipeline for the Mushroom Classification project."""

import json
import logging
import os
import sys
from datetime import datetime

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.model_registry import MushroomModelRegistry
from src.preprocess import build_dataset_metadata, load_data, run_preprocessing, save_processed_data
from src.train import cross_validate_models, get_baseline_models, save_model, train_baseline_models, tune_decision_tree, tune_random_forest
from src.evaluate import evaluate_model, evaluate_multiple_models, get_classification_report, get_confusion_matrix, get_feature_importances, save_evaluation_artifacts
from src.validate import save_validation_report, validate_dataset_schema, validate_preprocessed_data


def run_tracking_pipeline():
    logs_dir = os.path.join(PROJECT_ROOT, "logs")
    models_dir = os.path.join(PROJECT_ROOT, "models")
    outputs_dir = os.path.join(PROJECT_ROOT, "outputs")
    artifacts_dir = os.path.join(PROJECT_ROOT, "artifacts")
    os.makedirs(logs_dir, exist_ok=True)
    os.makedirs(models_dir, exist_ok=True)
    os.makedirs(outputs_dir, exist_ok=True)
    os.makedirs(artifacts_dir, exist_ok=True)

    log_file = os.path.join(logs_dir, "lab4_tracking.log")
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.FileHandler(log_file, mode="w", encoding="utf-8"),
            logging.StreamHandler(sys.stdout),
        ],
    )

    logging.info("=" * 60)
    logging.info("STARTING LAB 4: EXPERIMENT TRACKING & HYPERPARAMETER TUNING")
    logging.info("=" * 60)

    data_path = os.path.join(PROJECT_ROOT, "data", "raw", "mushroom.csv")
    raw_df = load_data(data_path)
    dataset_validation = validate_dataset_schema(raw_df)
    save_validation_report(dataset_validation, os.path.join(outputs_dir, "dataset_validation_report.json"))
    if not dataset_validation["valid"]:
        raise ValueError(f"Dataset validation failed: {dataset_validation['issues']}")

    metadata = build_dataset_metadata(raw_df)
    metadata_path = os.path.join(PROJECT_ROOT, "data", "processed", "dataset_metadata.json")
    os.makedirs(os.path.dirname(metadata_path), exist_ok=True)
    with open(metadata_path, "w", encoding="utf-8") as fh:
        json.dump(metadata, fh, indent=4)

    X_train, X_test, y_train, y_test = run_preprocessing(data_path=data_path)
    save_processed_data(X_train, X_test, y_train, y_test)
    preprocessing_validation = validate_preprocessed_data(X_train, X_test, y_train, y_test)
    save_validation_report(preprocessing_validation, os.path.join(outputs_dir, "preprocessing_validation_report.json"))
    if not preprocessing_validation["valid"]:
        raise ValueError(f"Preprocessing validation failed: {preprocessing_validation['issues']}")

    baseline_models = get_baseline_models()
    cv_df = cross_validate_models(baseline_models, X_train, y_train, cv=None, scoring="f1")
    logging.info("\nCross-validation results (F1):\n%s", cv_df.to_string(index=False))

    logging.info("\n--- Training tuned models ---")
    rf_grid = tune_random_forest(X_train, y_train, cv=5, scoring="f1")
    dt_grid = tune_decision_tree(X_train, y_train, cv=5, scoring="f1")
    best_rf = rf_grid.best_estimator_
    best_dt = dt_grid.best_estimator_

    logging.info("Best RF params: %s", rf_grid.best_params_)
    logging.info("Best RF CV F1: %.4f", rf_grid.best_score_)
    logging.info("Best DT params: %s", dt_grid.best_params_)
    logging.info("Best DT CV F1: %.4f", dt_grid.best_score_)

    trained_baselines = train_baseline_models(X_train, y_train, baseline_models)
    all_models = {
        **trained_baselines,
        "Tuned Random Forest": best_rf,
        "Tuned Decision Tree": best_dt,
    }
    final_comparison_df = evaluate_multiple_models(all_models, X_test, y_test)
    logging.info("\nFinal model comparison:\n%s", final_comparison_df.to_string(index=False))

    rf_metrics, _ = evaluate_model(best_rf, X_test, y_test)
    report = get_classification_report(best_rf, X_test, y_test)
    cm = get_confusion_matrix(best_rf, X_test, y_test)
    feature_importance = get_feature_importances(best_rf, X_train.columns, top_n=15)

    registry = MushroomModelRegistry(
        tracking_uri=os.path.join(PROJECT_ROOT, "artifacts", "mlruns"),
        experiment_name="mushroom-classification",
    )

    best_model_path = os.path.join(models_dir, "tuned_random_forest.joblib")
    save_model(best_rf, best_model_path)

    report_path = os.path.join(outputs_dir, "rf_classification_report.txt")
    confusion_path = os.path.join(outputs_dir, "rf_confusion_matrix.json")
    with open(report_path, "w", encoding="utf-8") as fh:
        fh.write(report)
    with open(confusion_path, "w", encoding="utf-8") as fh:
        json.dump(cm.tolist(), fh, indent=4)
    save_evaluation_artifacts(
        best_rf,
        X_test,
        y_test,
        outputs_dir,
        prefix="rf",
        artifact_dir=artifacts_dir,
    )

    run_metadata = {
        "timestamp": datetime.now().isoformat(),
        "dataset": metadata,
        "cross_validation_tracking": cv_df.to_dict(orient="records"),
        "random_forest_tuning": {"best_params": rf_grid.best_params_, "best_cv_f1": float(rf_grid.best_score_)},
        "decision_tree_tuning": {"best_params": dt_grid.best_params_, "best_cv_f1": float(dt_grid.best_score_)},
        "final_comparison": final_comparison_df.to_dict(orient="records"),
        "metrics": rf_metrics,
        "top_features": feature_importance.to_dict(orient="records"),
    }

    with open(os.path.join(outputs_dir, "tracking_results.json"), "w", encoding="utf-8") as fh:
        json.dump(run_metadata, fh, indent=4)

    registry.log_model_run(
        model_name="Random Forest",
        model=best_rf,
        params={
            "model_type": "RandomForestClassifier",
            "n_estimators": str(rf_grid.best_params_.get("n_estimators", 20)),
            "max_depth": str(rf_grid.best_params_.get("max_depth", 4)),
            "min_samples_split": str(rf_grid.best_params_.get("min_samples_split", 2)),
            "cv": "5",
            "scoring": "f1",
            "test_size": "0.20",
            "random_state": "42",
        },
        metrics=rf_metrics,
        dataset_metadata=metadata,
        preprocessing_metadata={
            "removed_constant_features": dataset_validation.get("constant_features", []),
            "feature_columns": list(X_train.columns),
            "target_mapping": {"e": 1, "p": 0},
        },
        validation_metadata=preprocessing_validation,
        artifact_file_paths={
            "classification_report": report_path,
            "confusion_matrix": confusion_path,
            "model_file": best_model_path,
        },
        classification_report=report,
        confusion_matrix=cm,
    )

    registry_summary = registry.get_registry_summary()
    with open(os.path.join(outputs_dir, "model_registry_summary.json"), "w", encoding="utf-8") as fh:
        json.dump(registry_summary, fh, indent=4)

    logging.info("\n[INFO] Tracking results saved to %s", os.path.join(outputs_dir, "tracking_results.json"))
    logging.info("[SUCCESS] Lab 4 Tracking Pipeline completed successfully!")
    return run_metadata


if __name__ == "__main__":
    run_tracking_pipeline()
