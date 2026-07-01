"""
src/intern4_fusion/calibration.py
====================================
Task 4.5: Post-hoc isotonic calibration applied to the final fused output.
"""

import numpy as np
from sklearn.isotonic import IsotonicRegression


def fit_isotonic_calibrators(probs_val: np.ndarray, labels_val: np.ndarray,
                              n_classes: int = 3) -> list:
    """Fit one isotonic regressor per class (one-vs-rest)."""
    calibrators = []
    for c in range(n_classes):
        iso = IsotonicRegression(out_of_bounds="clip")
        binary_labels = (labels_val == c).astype(float)
        iso.fit(probs_val[:, c], binary_labels)
        calibrators.append(iso)
    return calibrators


def apply_calibrators(probs: np.ndarray, calibrators: list) -> np.ndarray:
    """Apply per-class isotonic calibrators and re-normalise to sum to 1."""
    out = np.stack(
        [calibrators[c].predict(probs[:, c]) for c in range(len(calibrators))],
        axis=1,
    )
    row_sums = out.sum(axis=1, keepdims=True)
    out = out / np.where(row_sums == 0, 1, row_sums)
    return out.astype(np.float32)


def get_calibration_params(calibrators: list) -> dict:
    """Extract a JSON-serialisable summary of each calibrator's fit."""
    params = {}
    labels = ["normal", "mild", "severe"]
    for i, cal in enumerate(calibrators):
        params[labels[i]] = {
            "method": "isotonic_regression",
            "x_thresholds": [round(float(x), 4) for x in cal.X_thresholds_[:5]] + ["..."],
            "y_thresholds": [round(float(y), 4) for y in cal.y_thresholds_[:5]] + ["..."],
            "n_thresholds": int(len(cal.X_thresholds_)),
        }
    return params
