"""Preprocessing utilities for the Mushroom Classification project."""

import hashlib
import os
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


def get_default_data_path(filename: str = "data/raw/mushroom.csv") -> str:
    """Resolve the raw dataset path regardless of the current working directory."""
    candidates = [
        filename,
        os.path.join(PROJECT_ROOT, filename),
        os.path.join("..", filename),
        os.path.join(PROJECT_ROOT, "data", "raw", "mushroom.csv"),
    ]
    for candidate in candidates:
        if os.path.exists(candidate):
            return candidate
    return filename


def load_data(filepath: Optional[str] = None) -> pd.DataFrame:
    """Load the raw mushroom dataset."""
    data_path = get_default_data_path() if filepath is None else filepath
    return pd.read_csv(data_path)


def detect_constant_features(df: pd.DataFrame) -> List[str]:
    """Return columns with only one unique value."""
    return [col for col in df.columns if df[col].nunique() == 1]


def remove_constant_features(df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
    """Drop constant columns (e.g., veil-type) as in the notebook."""
    constant_features = detect_constant_features(df)
    if constant_features:
        df = df.drop(columns=constant_features)
    return df, constant_features


def separate_features_and_target(df: pd.DataFrame, target_col: str = "class") -> Tuple[pd.DataFrame, pd.Series]:
    """Separate X and y using the exact notebook logic."""
    drop_columns = [col for col in [target_col, "odor", "spore-print-color", "gill-color"] if col in df.columns]
    X = df.drop(columns=drop_columns)
    y = df[target_col].map({"e": 1, "p": 0})
    return X, y


def split_data(X: pd.DataFrame, y: pd.Series, test_size: float = 0.20, random_state: int = 42):
    """Split the data into train and test sets using the notebook settings."""
    return train_test_split(X, y, test_size=test_size, random_state=random_state)


def encode_features(X_train: pd.DataFrame, X_test: pd.DataFrame):
    """One-hot encode the categorical features and align the test data columns."""
    X_train_encoded = pd.get_dummies(X_train)
    X_test_encoded = pd.get_dummies(X_test)
    X_test_encoded = X_test_encoded.reindex(columns=X_train_encoded.columns, fill_value=0)
    return X_train_encoded, X_test_encoded


def build_dataset_metadata(df: pd.DataFrame, target_col: str = "class") -> Dict[str, object]:
    """Build a metadata dictionary from the actual Mushroom dataset."""
    dtypes = {col: str(df[col].dtype) for col in df.columns}
    missing_values = {col: int(df[col].isnull().sum()) for col in df.columns}
    categorical_features = [col for col in df.columns if df[col].dtype == "object" or pd.api.types.is_categorical_dtype(df[col])]
    dataset_hash = hashlib.sha256(pd.util.hash_pandas_object(df, index=True).values.tobytes()).hexdigest()
    constant_features = detect_constant_features(df)

    return {
        "rows": int(df.shape[0]),
        "columns": int(df.shape[1]),
        "column_names": list(df.columns),
        "target_column": target_col,
        "data_types": dtypes,
        "missing_values": missing_values,
        "categorical_features": categorical_features,
        "constant_features": constant_features,
        "duplicate_rows": int(df.duplicated().sum()),
        "class_distribution": {str(k): int(v) for k, v in df[target_col].value_counts().items()},
        "dataset_hash": dataset_hash,
    }


def run_preprocessing(data_path: Optional[str] = None, test_size: float = 0.20, random_state: int = 42, return_metadata: bool = False):
    """Execute the dataset cleaning and encoding flow used in the notebook."""
    print("[INFO] Loading dataset...")
    df = load_data(data_path)
    print(f"[INFO] Raw dataset shape: {df.shape}")

    print("[INFO] Removing constant features...")
    df_clean, removed_cols = remove_constant_features(df)
    if removed_cols:
        print(f"[INFO] Removed constant features: {removed_cols}")
    print(f"[INFO] Cleaned dataset shape: {df_clean.shape}")

    print("[INFO] Separating features and mapping target (e=1, p=0)...")
    X, y = separate_features_and_target(df_clean)

    print(f"[INFO] Splitting dataset (test_size={test_size}, random_state={random_state})...")
    X_train, X_test, y_train, y_test = split_data(X, y, test_size=test_size, random_state=random_state)

    print("[INFO] One-hot encoding categorical features...")
    X_train_encoded, X_test_encoded = encode_features(X_train, X_test)
    print(f"[INFO] Encoded training shape: {X_train_encoded.shape}")
    print(f"[INFO] Encoded testing shape: {X_test_encoded.shape}")

    if return_metadata:
        metadata = {
            "removed_constant_features": removed_cols,
            "raw_shape": list(df.shape),
            "cleaned_shape": list(df_clean.shape),
            "train_shape": list(X_train_encoded.shape),
            "test_shape": list(X_test_encoded.shape),
            "target_mapping": {"e": 1, "p": 0},
        }
        return X_train_encoded, X_test_encoded, y_train, y_test, metadata

    return X_train_encoded, X_test_encoded, y_train, y_test


def save_processed_data(X_train, X_test, y_train, y_test, output_dir=None):
    """Save the encoded train/test splits as NumPy arrays."""
    if output_dir is None:
        output_dir = os.path.join(PROJECT_ROOT, "data", "processed")
    os.makedirs(output_dir, exist_ok=True)

    arrays = {
        "X_train_final.npy": X_train.to_numpy(),
        "X_test_final.npy": X_test.to_numpy(),
        "y_train.npy": y_train.to_numpy(),
        "y_test.npy": y_test.to_numpy(),
    }
    for filename, array in arrays.items():
        np.save(os.path.join(output_dir, filename), array, allow_pickle=False)


if __name__ == "__main__":
    raw_df = load_data()
    metadata = build_dataset_metadata(raw_df)
    output_path = os.path.join(PROJECT_ROOT, "data", "processed", "dataset_metadata.json")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        import json
        json.dump(metadata, f, indent=4)
    print(f"[INFO] Wrote dataset metadata to {output_path}")
    X_train, X_test, y_train, y_test = run_preprocessing()
    save_processed_data(X_train, X_test, y_train, y_test)
    print("[SUCCESS] Preprocessing completed successfully.")
