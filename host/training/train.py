"""
train.py - load features, train LR / RF / GBT, compare, save best model.

Usage:
    python -m host.training.train
"""
import json
from pathlib import Path

import joblib
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

from host.pipeline.normalization import fit_and_save, load_and_transform
from host.training.cross_validation import day_based_cv
from host.training.evaluate import day_based_eval

DATA_PATH   = Path("data/processed/features.npz")
MODELS_DIR  = Path("models")


def main():
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Loading features from {DATA_PATH} ...")
    npz = np.load(DATA_PATH, allow_pickle=True)
    X      = npz["X"]
    y      = npz["y"]
    groups = npz["groups"]  # day strings per sample
    print(f"  X: {X.shape}  |  positive rate: {y.mean():.2%}")

    candidates = {
        "lr":  LogisticRegression(max_iter=1000, class_weight="balanced"),
        "rf":  RandomForestClassifier(n_estimators=200, class_weight="balanced", random_state=42),
        "gbt": GradientBoostingClassifier(n_estimators=200, max_depth=3, random_state=42),
    }

    print("\n--- Day-based LOGO cross-validation ---")
    cv_results = {}
    for name, clf in candidates.items():
        # day_based_cv fits/scales per-fold internally - pass raw features.
        result = day_based_cv(X, y, groups, clf)
        cv_results[name] = result
        print(f"  {name:>4}  AUC={result['mean_auc']:.3f} +/- {result['std_auc']:.3f}"
              f"  F1={result['mean_f1']:.3f} +/- {result['std_f1']:.3f}")

    # Pick best by mean AUC
    best_name = max(cv_results, key=lambda k: cv_results[k]["mean_auc"])
    print(f"\nBest model: {best_name}  (AUC={cv_results[best_name]['mean_auc']:.3f})")

    # --- Final model: train on all-days-except-last, evaluate on last day ---
    best_clf = candidates[best_name]
    test_day = np.unique(groups)[-1]  # matches day_based_eval's own default
    train_mask = groups != test_day

    # Fit the scaler on the train split only, then apply it to everything -
    # fitting on the full X would leak the held-out day into normalization.
    scaler_path = str(MODELS_DIR / f"{best_name}.scaler.joblib")
    fit_and_save(X[train_mask], scaler_path)
    X_scaled = load_and_transform(X, scaler_path)

    eval_metrics = day_based_eval(X_scaled, y, groups, best_clf, test_day=test_day)

    # Save fitted model
    model_path = MODELS_DIR / f"{best_name}.model.joblib"
    joblib.dump(best_clf, model_path)
    print(f"\nModel saved:  {model_path}")
    print(f"Scaler saved: {scaler_path}")

    # Save summary
    summary = {
        "best_model":  best_name,
        "cv_results":  cv_results,
        "eval_metrics": eval_metrics,
    }
    summary_path = MODELS_DIR / "training_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2))
    print(f"Summary:      {summary_path}")


if __name__ == "__main__":
    main()
