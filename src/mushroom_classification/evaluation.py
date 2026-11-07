from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


CLASS_NAMES = ["edible", "poisonous"]


def calculate_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_probability_poisonous: np.ndarray,
) -> dict[str, Any]:
    report = classification_report(
        y_true,
        y_pred,
        labels=[0, 1],
        target_names=CLASS_NAMES,
        output_dict=True,
        zero_division=0,
    )
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_precision": float(precision_score(y_true, y_pred, average="macro", zero_division=0)),
        "macro_recall": float(recall_score(y_true, y_pred, average="macro", zero_division=0)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "poisonous_precision": float(precision_score(y_true, y_pred, pos_label=1, zero_division=0)),
        "poisonous_recall": float(recall_score(y_true, y_pred, pos_label=1, zero_division=0)),
        "poisonous_f1": float(f1_score(y_true, y_pred, pos_label=1, zero_division=0)),
        "edible_recall": float(recall_score(y_true, y_pred, pos_label=0, zero_division=0)),
        "roc_auc_poisonous": float(roc_auc_score(y_true, y_probability_poisonous)),
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=[0, 1]).tolist(),
        "confusion_matrix_labels": CLASS_NAMES,
        "classification_report": report,
    }
