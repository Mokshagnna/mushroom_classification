"""Lab 3 baseline pipeline for the Mushroom Classification project."""

import json
import os
import sys

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.evaluate import evaluate_multiple_models, save_evaluation_artifacts
from src.preprocess import build_dataset_metadata, load_data, run_preprocessing, save_processed_data
from src.train import get_baseline_models, save_model, train_baseline_models
from src.validate import save_validation_report, validate_dataset_schema, validate_preprocessed_data


def run_baseline_pipeline():
    print("=" * 60)
    print("STARTING LAB 3: BASELINE CLASSIFICATION PIPELINE")
    print("=" * 60)

    data_path = os.path.join(PROJECT_ROOT, "data", "raw", "mushroom.csv")
    raw_df = load_data(data_path)

    dataset_validation = validate_dataset_schema(raw_df)
    save_validation_report(dataset_validation, os.path.join(PROJECT_ROOT, "outputs", "dataset_validation_report.json"))
    if not dataset_validation["valid"]:
        raise ValueError(f"Dataset validation failed: {dataset_validation['issues']}")

    metadata = build_dataset_metadata(raw_df)
    metadata_path = os.path.join(PROJECT_ROOT, "data", "processed", "dataset_metadata.json")
    os.makedirs(os.path.dirname(metadata_path), exist_ok=True)
    with open(metadata_path, "w", encoding="utf-8") as fh:
        json.dump(metadata, fh, indent=4)
    print(f"[INFO] Dataset metadata saved to {metadata_path}")

    X_train, X_test, y_train, y_test = run_preprocessing(data_path=data_path)
    save_processed_data(X_train, X_test, y_train, y_test)
    preprocessing_validation = validate_preprocessed_data(X_train, X_test, y_train, y_test)
    save_validation_report(preprocessing_validation, os.path.join(PROJECT_ROOT, "outputs", "preprocessing_validation_report.json"))
    if not preprocessing_validation["valid"]:
        raise ValueError(f"Preprocessing validation failed: {preprocessing_validation['issues']}")

    print("\n[INFO] Initializing and training baseline models...")
    models = get_baseline_models()
    trained_models = train_baseline_models(X_train, y_train, models)

    print("\n[INFO] Evaluating baseline models on test set...")
    results_df = evaluate_multiple_models(trained_models, X_test, y_test)
    print("\nBaseline model comparison:")
    print(results_df.to_string(index=False))

    best_model_name = results_df.iloc[0]["Model"]
    best_model = trained_models[best_model_name]
    artifact_dir = os.path.join(PROJECT_ROOT, "outputs")
    saved_artifacts = save_evaluation_artifacts(
        best_model,
        X_test,
        y_test,
        artifact_dir,
        prefix="baseline_rf",
        artifact_dir=os.path.join(PROJECT_ROOT, "artifacts"),
    )

    models_dir = os.path.join(PROJECT_ROOT, "models")
    os.makedirs(models_dir, exist_ok=True)
    save_model(best_model, os.path.join(models_dir, "baseline_best_model.joblib"))

    metrics_save_path = os.path.join(PROJECT_ROOT, "outputs", "baseline_metrics.json")
    with open(metrics_save_path, "w", encoding="utf-8") as fh:
        json.dump(
            {
                "pipeline": "Lab 3 Baseline Pipeline",
                "best_model": best_model_name,
                "model_comparison": results_df.to_dict(orient="records"),
                "metrics": saved_artifacts["metrics"],
                "confusion_matrix": saved_artifacts["confusion_matrix"].tolist(),
            },
            fh,
            indent=4,
        )

    print(f"[INFO] Baseline metrics saved to {metrics_save_path}")
    print("\n[SUCCESS] Lab 3 Baseline Pipeline executed successfully!")
    return results_df


if __name__ == "__main__":
    run_baseline_pipeline()
