"""Evaluation utilities for the Mushroom Classification project."""

import json
import os
from typing import Dict, Iterable

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    RocCurveDisplay,
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)


def compute_metrics(y_true, y_pred):
    """Compute the notebook metrics: accuracy, precision, recall, and F1."""
    return {
        "Accuracy": float(accuracy_score(y_true, y_pred)),
        "Precision": float(precision_score(y_true, y_pred)),
        "Recall": float(recall_score(y_true, y_pred)),
        "F1 Score": float(f1_score(y_true, y_pred)),
    }


def evaluate_model(model, X_test, y_test):
    """Evaluate a single trained model and return metrics plus predictions."""
    y_pred = model.predict(X_test)
    return compute_metrics(y_test, y_pred), y_pred


def evaluate_multiple_models(models_dict, X_test, y_test):
    """Evaluate a model dictionary and return a DataFrame sorted by F1 score."""
    results = []
    for name, model in models_dict.items():
        metrics, _ = evaluate_model(model, X_test, y_test)
        metrics["Model"] = name
        results.append(metrics)

    df = pd.DataFrame(results)
    columns = ["Model", "Accuracy", "Precision", "Recall", "F1 Score"]
    return df[columns].sort_values(by="F1 Score", ascending=False).reset_index(drop=True)


def get_confusion_matrix(model, X_test, y_test):
    """Compute the confusion matrix for a trained model."""
    y_pred = model.predict(X_test)
    return confusion_matrix(y_test, y_pred)


def get_classification_report(model, X_test, y_test, output_dict=False):
    """Return the detailed classification report from the notebook."""
    y_pred = model.predict(X_test)
    return classification_report(y_test, y_pred, output_dict=output_dict)


def get_feature_importances(model, feature_names, top_n=15):
    """Return the feature importance table for models with feature_importances_."""
    if not hasattr(model, "feature_importances_"):
        raise AttributeError(f"Model {type(model).__name__} does not expose feature_importances_")

    importance_df = pd.DataFrame({
        "Feature": list(feature_names),
        "Importance": model.feature_importances_,
    })
    importance_df = importance_df.sort_values(by="Importance", ascending=False).reset_index(drop=True)
    if top_n is not None:
        return importance_df.head(top_n)
    return importance_df


def save_evaluation_artifacts(model, X_test, y_test, output_dir, prefix="model", artifact_dir=None):
    """Persist evaluation artifacts to the output folder."""
    os.makedirs(output_dir, exist_ok=True)
    if artifact_dir is None:
        artifact_dir = os.path.join(os.path.dirname(output_dir), "artifacts")
    os.makedirs(artifact_dir, exist_ok=True)

    metrics, y_pred = evaluate_model(model, X_test, y_test)
    report = get_classification_report(model, X_test, y_test, output_dict=True)
    cm = get_confusion_matrix(model, X_test, y_test)

    report_path = os.path.join(output_dir, f"{prefix}_classification_report.txt")
    confusion_path = os.path.join(output_dir, f"{prefix}_confusion_matrix.json")
    metrics_path = os.path.join(output_dir, f"{prefix}_metrics.json")
    false_positive_path = os.path.join(output_dir, "false_positives.csv")
    false_negative_path = os.path.join(output_dir, "false_negatives.csv")
    confusion_plot_path = os.path.join(artifact_dir, "confusion_matrix.png")
    roc_plot_path = os.path.join(artifact_dir, "roc_curve.png")

    misclassified = X_test.copy() if isinstance(X_test, pd.DataFrame) else pd.DataFrame(X_test)
    if not isinstance(X_test, pd.DataFrame) and hasattr(y_test, "index"):
        misclassified.index = y_test.index
    misclassified["y_true"] = list(y_test)
    misclassified["y_pred"] = list(y_pred)
    misclassified.index.name = "row_index"
    misclassified.loc[(misclassified["y_true"] == 0) & (misclassified["y_pred"] == 1)].to_csv(false_positive_path)
    misclassified.loc[(misclassified["y_true"] == 1) & (misclassified["y_pred"] == 0)].to_csv(false_negative_path)

    figure, axis = plt.subplots()
    ConfusionMatrixDisplay.from_predictions(y_test, y_pred, ax=axis)
    figure.savefig(confusion_plot_path, bbox_inches="tight")
    plt.close(figure)

    figure, axis = plt.subplots()
    RocCurveDisplay.from_estimator(model, X_test, y_test, ax=axis)
    figure.savefig(roc_plot_path, bbox_inches="tight")
    plt.close(figure)

    with open(report_path, "w", encoding="utf-8") as fh:
        fh.write(get_classification_report(model, X_test, y_test, output_dict=False))

    with open(confusion_path, "w", encoding="utf-8") as fh:
        json.dump(cm.tolist(), fh, indent=4)

    with open(metrics_path, "w", encoding="utf-8") as fh:
        json.dump(metrics, fh, indent=4)

    return {
        "metrics": metrics,
        "classification_report": report,
        "confusion_matrix": cm,
        "report_path": report_path,
        "confusion_path": confusion_path,
        "metrics_path": metrics_path,
        "false_positive_path": false_positive_path,
        "false_negative_path": false_negative_path,
        "confusion_plot_path": confusion_plot_path,
        "roc_plot_path": roc_plot_path,
    }
