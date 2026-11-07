from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib

from src.mushroom_classification.config import get_model_path
from src.mushroom_classification.validation import FEATURE_COLUMNS


def load_model(model_path: Path | None = None) -> Any:
    path = model_path or get_model_path()
    if not path.is_file():
        raise FileNotFoundError(
            f"Trained mushroom model not found: {path}. Run the training pipeline first."
        )
    model = joblib.load(path)
    if not callable(getattr(model, "predict", None)) or not callable(
        getattr(model, "predict_proba", None)
    ):
        raise ValueError(f"Saved artifact is not a prediction pipeline: {path}")
    classes = set(getattr(model, "classes_", ()))
    if classes != {0, 1}:
        raise ValueError(
            "Saved mushroom model must contain both classes: edible=0 and poisonous=1."
        )
    model_features = getattr(model, "feature_names_in_", None)
    if model_features is not None and tuple(model_features) != FEATURE_COLUMNS:
        raise ValueError("Saved mushroom model does not match the 22-feature schema.")
    return model


class ModelLoader:
    def __init__(self, model_path: Path | None = None) -> None:
        self.model_path = model_path or get_model_path()
        self.model: Any | None = None

    def load(self) -> Any:
        self.model = load_model(self.model_path)
        return self.model

    def is_ready(self) -> bool:
        return self.model is not None
