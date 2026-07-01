"""
src/intern4_fusion/export.py
===============================
Task 4.6: Export unified_score.json using the EXACT schema from
the spec document (Section 5.2, Task 4.6):

{
  "patient_id": str,
  "timestamp": ISO-8601,
  "final_predictions": {"normal": float, "mild": float, "severe": float},
  "calibrated_probabilities": {"normal": float, "mild": float, "severe": float},
  "fused_risk_score": int,
  "meta_learner_weights": List[float],
  "calibration_params": dict,
  "model_ensemble_version": str
}
"""

import json
import numpy as np
from pathlib import Path
from datetime import datetime, timezone

SEVERITY_LABELS = ["normal", "mild", "severe"]


def export_unified_scores(patient_ids: list,
                           final_probs: np.ndarray,
                           calibrated_probs: np.ndarray,
                           meta_learner_weights: list,
                           calibration_params: dict,
                           out_dir: Path,
                           model_ensemble_version: str = "v1.0_late+xgb+isotonic",
                           prefer_uncalibrated_class: bool = False):
    """
    Write one unified_score.json per patient, matching the spec schema exactly.

    prefer_uncalibrated_class: if True, fused_risk_score and the implicit
        class decision are derived from `final_probs` (pre-calibration)
        rather than `calibrated_probs`. Use this when isotonic calibration
        was found to regress balanced accuracy on a thin validation fold —
        the calibrated_probabilities field is still reported for
        confidence/ECE purposes, but it does not drive the risk score used
        downstream by Intern 5.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now(timezone.utc).isoformat()

    risk_source = final_probs if prefer_uncalibrated_class else calibrated_probs

    for i, pid in enumerate(patient_ids):
        fused_risk_score = int(round(
            (risk_source[i, 1] + risk_source[i, 2]) * 100
        ))

        payload = {
            "patient_id": pid,
            "timestamp": ts,
            "final_predictions": {
                "normal": round(float(final_probs[i, 0]), 4),
                "mild": round(float(final_probs[i, 1]), 4),
                "severe": round(float(final_probs[i, 2]), 4),
            },
            "calibrated_probabilities": {
                "normal": round(float(calibrated_probs[i, 0]), 4),
                "mild": round(float(calibrated_probs[i, 1]), 4),
                "severe": round(float(calibrated_probs[i, 2]), 4),
            },
            "fused_risk_score": fused_risk_score,
            "meta_learner_weights": [round(float(w), 4) for w in meta_learner_weights],
            "calibration_params": calibration_params,
            "model_ensemble_version": model_ensemble_version,
        }

        with open(out_dir / f"{pid}.json", "w") as fh:
            json.dump(payload, fh, indent=2)

    print(f"Exported {len(patient_ids)} unified_score.json files to {out_dir}/")


def validate_schema(sample_path: Path):
    """Quick self-check that a written file matches the required schema."""
    data = json.load(open(sample_path))
    required_keys = {
        "patient_id", "timestamp", "final_predictions",
        "calibrated_probabilities", "fused_risk_score",
        "meta_learner_weights", "calibration_params",
        "model_ensemble_version",
    }
    missing = required_keys - set(data.keys())
    assert not missing, f"Schema validation FAILED — missing keys: {missing}"

    for key in ["final_predictions", "calibrated_probabilities"]:
        s = sum(data[key].values())
        assert abs(s - 1.0) < 0.01, f"{key} does not sum to 1.0 (got {s})"

    assert 0 <= data["fused_risk_score"] <= 100, "fused_risk_score out of range"
    print(f"Schema validation PASSED for {sample_path.name}")
    return data
