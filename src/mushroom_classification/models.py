from __future__ import annotations

from typing import Any

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.tree import DecisionTreeClassifier

from .validation import FEATURE_COLUMNS


def build_preprocessor() -> ColumnTransformer:
    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="constant", fill_value="__missing__")),
            ("one_hot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )
    return ColumnTransformer(
        transformers=[("categorical", categorical_pipeline, list(FEATURE_COLUMNS))],
        remainder="drop",
    )


def build_classifier(name: str, random_state: int, n_estimators: int) -> Any:
    if name == "Logistic Regression":
        return LogisticRegression(max_iter=1000)
    if name == "Decision Tree":
        return DecisionTreeClassifier(random_state=random_state)
    if name == "Random Forest":
        return RandomForestClassifier(
            n_estimators=n_estimators,
            random_state=random_state,
            n_jobs=1,
        )
    if name == "K-Nearest Neighbors":
        return KNeighborsClassifier(n_neighbors=5)
    raise ValueError(f"Unsupported model candidate: {name}")


def build_model_pipeline(name: str, random_state: int, n_estimators: int) -> Pipeline:
    return Pipeline(
        steps=[
            ("preprocessor", build_preprocessor()),
            ("classifier", build_classifier(name, random_state, n_estimators)),
        ]
    )
