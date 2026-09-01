"""
clean_segment_features.py - beat detection, artifact rejection, and HR
trend / HRV feature computation from the "clean early segment" of a 180s
lookback window.

Clean early segment: the first 90 seconds of the 180s window
([end_ms-180000, end_ms-90000) in absolute terms). That leaves a ~97s
total gap to event onset once the 7s event buffer is added - comfortably
clear of motion artifact near the event. CLEAN_SEGMENT_MS is adjustable.

Beat detection mirrors host/pipeline/features.py's _ppg_hr_features
(0.7-3.5Hz bandpass, find_peaks with a 220bpm-ceiling minimum distance),
reimplemented here rather than imported since that function only returns a
summary dict, not the peak positions needed for trend fitting and artifact
rejection.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt, find_peaks

CLEAN_SEGMENT_MS = 90_000  # first 90s of the 180s window; adjustable

# Same cardiac-band + peak-distance parameters as host/pipeline/features.py's
# _ppg_hr_features, for methodological consistency.
_PPG_BAND_LOW_HZ = 0.7
_PPG_BAND_HIGH_HZ = 3.5
_PPG_MAX_BPM = 220
_FS = 50.0  # matches host/pipeline/features.py's _FS

# Ectopic-beat / artifact rejection: reject an IBI if it differs from the
# LOCAL median (a rolling window of neighboring beats, not the whole
# segment) by more than this fraction - a standard, simple ectopic-beat
# heuristic (comparable to the ~20% rule used in several clinical HRV
# artifact-correction schemes).
LOCAL_MEDIAN_WINDOW = 5      # beats on each side
REJECT_FRAC_THRESHOLD = 0.20

# Minimum valid data required before trusting a feature at all, rather than
# emitting NaN. Beat *count* matters more here than segment duration, since
# rejection can hollow out an otherwise long segment.
MIN_VALID_BEATS_FOR_TREND = 5
MIN_VALID_IBIS_FOR_HRV = 10
PLAUSIBLE_HR_RANGE_BPM = (40, 180)  # outside this, treat detection as having locked onto noise


def detect_beat_peaks(ppg_ir: np.ndarray, fs: float = _FS) -> np.ndarray:
    """Sample-index positions of detected cardiac beats. Empty array if the
    segment is too short or the filter can't be applied."""
    n = len(ppg_ir)
    if n < 4:
        return np.array([], dtype=int)
    nyq = fs / 2.0
    low, high = _PPG_BAND_LOW_HZ / nyq, min(_PPG_BAND_HIGH_HZ / nyq, 0.99)
    b, a = butter(2, [low, high], btype="band")
    try:
        filtered = filtfilt(b, a, ppg_ir)
    except ValueError:
        return np.array([], dtype=int)
    min_distance = max(1, int(fs * 60.0 / _PPG_MAX_BPM))
    peaks, _ = find_peaks(filtered, distance=min_distance)
    return peaks


def reject_ectopic_ibis(ibi_ms: np.ndarray) -> np.ndarray:
    """Boolean valid-mask over ibi_ms: True = kept. Rejects any IBI more
    than REJECT_FRAC_THRESHOLD away from the median of its
    LOCAL_MEDIAN_WINDOW neighbors on each side."""
    n = len(ibi_ms)
    valid = np.ones(n, dtype=bool)
    for i in range(n):
        lo, hi = max(0, i - LOCAL_MEDIAN_WINDOW), min(n, i + LOCAL_MEDIAN_WINDOW + 1)
        neighborhood = np.delete(ibi_ms[lo:hi], i - lo)
        if len(neighborhood) == 0:
            continue
        local_median = np.median(neighborhood)
        if local_median <= 0:
            continue
        if abs(ibi_ms[i] - local_median) / local_median > REJECT_FRAC_THRESHOLD:
            valid[i] = False
    return valid


def clean_segment_cardiac_features(window_df: pd.DataFrame, end_ms: int) -> dict:
    """Extract the clean early segment from a 180s window (window_df must
    already BE that 180s window, indexed by timestamp_ms) and compute HR
    trend + HRV features from it.

    Returns a dict with the 4 features (hr_trend_mean, hr_trend_slope,
    rmssd_clean, sdnn_clean) - NaN for any that fail their quality gate -
    plus a 'quality' sub-dict recording exactly why, for reporting.
    """
    seg_start = end_ms - 180_000
    seg_end = seg_start + CLEAN_SEGMENT_MS
    segment = window_df.loc[seg_start:seg_end - 1]

    result = {
        "hr_trend_mean": float("nan"), "hr_trend_slope": float("nan"),
        "rmssd_clean": float("nan"), "sdnn_clean": float("nan"),
    }
    quality = {
        "n_samples_in_segment": len(segment), "n_raw_peaks": 0, "raw_hr_mean_bpm": float("nan"),
        "plausible_hr": False, "n_valid_ibis": 0, "trend_ok": False, "hrv_ok": False,
    }

    if len(segment) < 10:
        return {"features": result, "quality": quality}

    ppg_ir = segment["ppg_ir"].to_numpy(dtype=float)
    peaks = detect_beat_peaks(ppg_ir, fs=_FS)
    quality["n_raw_peaks"] = len(peaks)
    if len(peaks) < 2:
        return {"features": result, "quality": quality}

    peak_times_ms = segment.index.to_numpy()[peaks].astype(float)
    ibi_ms = np.diff(peak_times_ms)  # use actual timestamps, not fs-derived spacing - robust to sample gaps
    ibi_ms = ibi_ms[ibi_ms > 0]
    if len(ibi_ms) < 2:
        return {"features": result, "quality": quality}

    raw_hr = 60000.0 / ibi_ms
    quality["raw_hr_mean_bpm"] = float(np.mean(raw_hr))
    lo, hi = PLAUSIBLE_HR_RANGE_BPM
    if not (lo <= quality["raw_hr_mean_bpm"] <= hi):
        # Peak detection likely locked onto motion noise, not real beats -
        # don't bother computing anything further, the whole segment's
        # detected "beats" aren't cardiac in origin.
        return {"features": result, "quality": quality}
    quality["plausible_hr"] = True

    valid_mask = reject_ectopic_ibis(ibi_ms)
    quality["n_valid_ibis"] = int(valid_mask.sum())
    valid_ibi = ibi_ms[valid_mask]
    valid_peak_times = peak_times_ms[1:][valid_mask]  # time of the beat ending each valid IBI

    # HR trend: fit HR(t) = slope*t + intercept over valid beats only.
    if len(valid_ibi) >= MIN_VALID_BEATS_FOR_TREND:
        hr_vals = 60000.0 / valid_ibi
        t_sec = (valid_peak_times - seg_start) / 1000.0
        slope, intercept = np.polyfit(t_sec, hr_vals, deg=1)
        result["hr_trend_mean"] = float(np.mean(hr_vals))
        result["hr_trend_slope"] = float(slope * 60.0)  # bpm per minute, more interpretable
        quality["trend_ok"] = True

    # HRV: RMSSD needs ADJACENT valid pairs (never diff across a rejected
    # gap); SDNN just needs the set of valid IBIs.
    if len(valid_ibi) >= MIN_VALID_IBIS_FOR_HRV:
        adjacent_valid = valid_mask[:-1] & valid_mask[1:]
        successive_diffs = np.diff(ibi_ms)[adjacent_valid]
        if len(successive_diffs) >= MIN_VALID_IBIS_FOR_HRV - 1:
            result["rmssd_clean"] = float(np.sqrt(np.mean(successive_diffs ** 2)))
        result["sdnn_clean"] = float(np.std(valid_ibi))
        quality["hrv_ok"] = not np.isnan(result["rmssd_clean"]) and not np.isnan(result["sdnn_clean"])

    return {"features": result, "quality": quality}
