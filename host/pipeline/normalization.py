"""
normalization.py - fit/save and load/transform StandardScaler.

Usage pattern:
    # At training time:
    scaler_path = "models/best_model.scaler.joblib"
    X_scaled = fit_and_save(X_train, scaler_path)

    # At inference time:
    X_scaled = load_and_transform(X_live, scaler_path)
"""
import joblib
import numpy as np
from sklearn.preprocessing import StandardScaler


def fit_and_save(X: np.ndarray, path: str) -> np.ndarray:
    """Fit a StandardScaler on X, save it to *path*, and return X_scaled."""
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    joblib.dump(scaler, path)
    return X_scaled


def load_and_transform(X: np.ndarray, path: str) -> np.ndarray:
    """Load a previously saved scaler and apply it to X."""
    scaler: StandardScaler = joblib.load(path)
    return scaler.transform(X)
