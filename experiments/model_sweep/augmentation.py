"""
augmentation.py - raw-signal augmentation (jitter, magnitude
warping, time warping), applied ONLY to training-fold raw window data
before feature extraction. Augmented copies inherit their source window's
day label and are generated fresh inside augmented_day_cv's per-fold loop,
so they can never appear in a test fold - the held-out day is always
scored on its real, unaugmented windows only.

Standard HAR augmentation techniques (see e.g. Um et al. 2017):
  - jitter: additive Gaussian noise, small relative to each channel's own std.
  - magnitude_warp: multiply the signal by a smooth random curve (cubic
    spline through a few random control points around 1.0) - simulates
    sensor gain variation / slightly different force application.
  - time_warp: smoothly stretch/compress the time axis, then resample back
    onto the original timestamps - simulates the same movement performed
    slightly faster/slower.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.interpolate import CubicSpline
from sklearn.base import clone
from sklearn.metrics import f1_score, roc_auc_score
from sklearn.preprocessing import StandardScaler

ACCEL_GYRO_COLS = ("accel_x", "accel_y", "accel_z", "gyro_x", "gyro_y", "gyro_z")

JITTER_STD_FRAC = 0.02
WARP_N_KNOTS = 4
WARP_STD = 0.1


def jitter(df: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    aug = df.copy()
    for col in ACCEL_GYRO_COLS:
        std = df[col].to_numpy(dtype=float).std() or 1.0
        aug[col] = df[col] + rng.normal(0, JITTER_STD_FRAC * std, size=len(df))
    return aug


def magnitude_warp(df: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    n = len(df)
    knot_x = np.linspace(0, n - 1, WARP_N_KNOTS)
    knot_y = rng.normal(1.0, WARP_STD, size=WARP_N_KNOTS)
    curve = CubicSpline(knot_x, knot_y)(np.arange(n))
    aug = df.copy()
    for col in ACCEL_GYRO_COLS:
        aug[col] = df[col].to_numpy(dtype=float) * curve
    return aug


def time_warp(df: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    n = len(df)
    if n < WARP_N_KNOTS + 1:
        return df.copy()
    knot_x = np.linspace(0, n - 1, WARP_N_KNOTS)
    knot_y = np.cumsum(rng.normal(1.0, WARP_STD, size=WARP_N_KNOTS))
    knot_y = (knot_y - knot_y.min()) / (knot_y.max() - knot_y.min()) * (n - 1)  # monotonic, spans [0, n-1]
    warp_curve = CubicSpline(knot_x, knot_y)(np.arange(n))
    warp_curve = np.clip(np.sort(warp_curve), 0, n - 1)  # enforce monotonicity after spline

    orig_idx = np.arange(n)
    aug = df.copy()
    for col in ACCEL_GYRO_COLS:
        vals = df[col].to_numpy(dtype=float)
        aug[col] = np.interp(orig_idx, warp_curve, vals)
    return aug


AUGMENTATIONS = {"jitter": jitter, "magnitude_warp": magnitude_warp, "time_warp": time_warp}


def augmented_day_cv(windows: list, clf_factory, feature_fn, augmentation_names: list[str],
                      n_augments_per_window: int = 1, seed: int = 42) -> dict:
    """Day-based LOGO; training-fold windows get n_augments_per_window
    extra augmented copies per listed augmentation (applied to raw signal,
    then feature_fn is called fresh on each copy) before fitting. Test-fold
    windows are always the real, unaugmented data. clf_factory() must
    return a fresh unfitted sklearn-compatible classifier each call.
    """
    from sklearn.model_selection import LeaveOneGroupOut

    days = sorted({w.day_str for w in windows})
    rng = np.random.default_rng(seed)
    aucs, f1s = [], []

    for held_out_day in days:
        train_windows = [w for w in windows if w.day_str != held_out_day]
        test_windows = [w for w in windows if w.day_str == held_out_day]
        if not train_windows or not test_windows:
            continue

        feature_names = None
        X_train_rows, y_train = [], []
        for w in train_windows:
            copies = [w.df]
            for aug_name in augmentation_names:
                aug_fn = AUGMENTATIONS[aug_name]
                copies.extend(aug_fn(w.df, rng) for _ in range(n_augments_per_window))
            for copy_df in copies:
                feats = feature_fn(copy_df)
                if feature_names is None:
                    feature_names = sorted(feats.keys())
                X_train_rows.append([feats[k] for k in feature_names])
                y_train.append(w.label)

        X_test_rows, y_test = [], []
        for w in test_windows:
            feats = feature_fn(w.df)
            X_test_rows.append([feats[k] for k in feature_names])
            y_test.append(w.label)

        X_train = np.array(X_train_rows, dtype=float)
        X_test = np.array(X_test_rows, dtype=float)
        y_train_arr, y_test_arr = np.array(y_train), np.array(y_test)
        if len(np.unique(y_test_arr)) < 2:
            continue

        all_nan = np.isnan(X_train).all(axis=0)
        X_train, X_test = X_train[:, ~all_nan], X_test[:, ~all_nan]
        col_medians = np.nanmedian(X_train, axis=0)
        for arr in (X_train, X_test):
            mask = np.isnan(arr)
            if mask.any():
                r, c = np.where(mask)
                arr[r, c] = col_medians[c]

        scaler = StandardScaler()
        X_train_s, X_test_s = scaler.fit_transform(X_train), scaler.transform(X_test)

        clf = clone(clf_factory())
        clf.fit(X_train_s, y_train_arr)
        y_prob = clf.predict_proba(X_test_s)[:, 1]
        y_pred = clf.predict(X_test_s)
        aucs.append(roc_auc_score(y_test_arr, y_prob))
        f1s.append(f1_score(y_test_arr, y_pred, zero_division=0))

    aucs_arr, f1s_arr = np.array(aucs), np.array(f1s)
    return {
        "mean_auc": float(aucs_arr.mean()) if len(aucs_arr) else float("nan"),
        "std_auc":  float(aucs_arr.std())  if len(aucs_arr) else float("nan"),
        "mean_f1":  float(f1s_arr.mean())  if len(f1s_arr) else float("nan"),
        "std_f1":   float(f1s_arr.std())   if len(f1s_arr) else float("nan"),
        "fold_aucs": aucs_arr.tolist(),
        "fold_f1s":  f1s_arr.tolist(),
    }
