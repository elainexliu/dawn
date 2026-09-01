"""Fits and saves a deployable model/scaler for live testing: D @ 3s buffer
+ clean-segment HRV, lr_l2_baseline - the only statistically-confirmed
candidate from this experiment (see results.md). Full-data fit for live
testing only, not a held-out evaluation; the CV/bootstrap evidence lives in
results.md, not here. Not merged into host/training/ - same standing
recommendation as the rest of this experiment.

Usage:
    python -m experiments.best_model_followup.train_deployable
"""
from pathlib import Path

import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from experiments.best_model_followup.run_experiment import add_hrv_matched, build_feature_set

OUT_DIR = Path(__file__).parent


def main():
    X, y, groups, names, anchors = build_feature_set(buffer_ms=3000, which="D")
    _, X_hrv, y_c, groups_c, ok_mask = add_hrv_matched(X, y, groups, anchors)
    feature_names = names + ["rmssd_clean", "sdnn_clean"]

    print(f"D@3000+hrv: n={X_hrv.shape[0]}  features={X_hrv.shape[1]}  "
          f"positive_rate={y_c.mean():.2%}  days={sorted(set(groups_c))}")

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_hrv)
    model = LogisticRegression(C=0.01, max_iter=2000, class_weight="balanced").fit(X_scaled, y_c)

    joblib.dump(model, OUT_DIR / "d3s_hrv_lr.model.joblib")
    joblib.dump(scaler, OUT_DIR / "d3s_hrv_lr.scaler.joblib")
    np.savez(OUT_DIR / "d3s_hrv_features.npz",
             X=X_hrv, feature_names=np.array(feature_names), y=y_c, groups=groups_c)

    print(f"Saved: {OUT_DIR / 'd3s_hrv_lr.model.joblib'}")
    print(f"Saved: {OUT_DIR / 'd3s_hrv_lr.scaler.joblib'}")
    print(f"Saved: {OUT_DIR / 'd3s_hrv_features.npz'}")


if __name__ == "__main__":
    main()
