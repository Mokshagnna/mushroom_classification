"""Training and hyperparameter-tuning logic for the Mushroom Classification project."""

import os
from typing import Dict, Iterable, Optional

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier


def get_baseline_models() -> Dict[str, object]:
    """Return the exact model set used in the notebook."""
    return {
        "Logistic Regression": LogisticRegression(max_iter=1000),
        "Decision Tree": DecisionTreeClassifier(max_depth=4, random_state=42),
        "Random Forest": RandomForestClassifier(n_estimators=20, max_depth=4, random_state=42),
        "KNN": KNeighborsClassifier(n_neighbors=35),
    }


def train_model(model, X_train, y_train):
    """Fit a single model on the training split."""
    model.fit(X_train, y_train)
    return model


def train_baseline_models(X_train, y_train, models=None):
    """Train all baseline models and return them in a dictionary."""
    if models is None:
        models = get_baseline_models()

    trained_models = {}
    for name, model in models.items():
        print(f"[INFO] Training {name}...")
        model.fit(X_train, y_train)
        trained_models[name] = model
    return trained_models


def cross_validate_models(models, X_train, y_train, cv=None, scoring="f1"):
    """Run stratified cross-validation and return a DataFrame of CV summaries."""
    if cv is None:
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    cv_results = []
    for name, model in models.items():
        scores = cross_val_score(model, X_train, y_train, cv=cv, scoring=scoring)
        cv_results.append(
            {
                "Model": name,
                "Mean CV Score": float(scores.mean()),
                "Std CV Score": float(scores.std()),
            }
        )

    cv_df = pd.DataFrame(cv_results)
    return cv_df.sort_values(by="Mean CV Score", ascending=False).reset_index(drop=True)


def tune_random_forest(X_train, y_train, cv=5, scoring="f1", n_jobs=-1):
    """Apply GridSearchCV to the Random Forest configuration used in the notebook."""
    rf = RandomForestClassifier(random_state=42)
    rf_params = {
        "n_estimators": [10, 20],
        "max_depth": [3, 4, 5],
        "min_samples_split": [2, 5],
    }
    rf_grid = GridSearchCV(rf, rf_params, cv=cv, scoring=scoring, n_jobs=n_jobs)
    rf_grid.fit(X_train, y_train)
    return rf_grid


def tune_decision_tree(X_train, y_train, cv=5, scoring="f1", n_jobs=-1):
    """Apply GridSearchCV to the Decision Tree configuration used in the notebook."""
    dt = DecisionTreeClassifier(random_state=42)
    dt_params = {
        "max_depth": [2, 3],
        "min_samples_split": [10, 20],
        "min_samples_leaf": [5, 10],
    }
    dt_grid = GridSearchCV(dt, dt_params, cv=cv, scoring=scoring, n_jobs=n_jobs)
    dt_grid.fit(X_train, y_train)
    return dt_grid


def save_model(model, filepath):
    """Persist a trained model to disk."""
    os.makedirs(os.path.dirname(filepath) or ".", exist_ok=True)
    joblib.dump(model, filepath)
    print(f"[INFO] Model saved to {filepath}")


def load_model(filepath):
    """Load a previously saved model."""
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Model file not found at {filepath}")
    return joblib.load(filepath)
