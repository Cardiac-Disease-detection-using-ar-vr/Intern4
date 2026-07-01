"""
src/intern4_fusion/data_loader.py
==================================
Loads and aligns data from Intern 2 (Vasanth) and Intern 3 (Tushar).

CRITICAL FIX — Mild class collapse:
  Tushar's `class_predictions.mild` is capped near 0.15 and NEVER wins
  argmax across all 1026 patients (verified empirically). His underlying
  model is effectively binary (healthy vs disease); the "mild" column is
  not a trained, usable signal.

  Fix: reconstruct proper 3-class probabilities from two signals that
  ARE meaningful in his outputs:
    - common_alpha   (Dempster-Shafer fused disease probability, parquet)
    - ef_prediction  (continuous EF estimate, JSON)

  P(Normal) = 1 - common_alpha
  P(disease) is split into Mild vs Severe using a sigmoid centered at
  EF=40 (the clinical Mild/Severe boundary), so:
    EF >> 40  -> mostly Mild
    EF << 40  -> mostly Severe
"""

import json
import glob
import pickle
import numpy as np
import pandas as pd
from pathlib import Path
from collections import Counter


def pat_num(pid: str) -> int:
    return int(pid.split("_")[1])


# ─────────────────────────────────────────────
# VASANTH (Intern 2) — Base Module
# ─────────────────────────────────────────────

def load_vasanth_data(base_scores_dir: Path) -> dict:
    files = sorted(glob.glob(str(base_scores_dir / "*_base.json")))
    assert len(files) > 0, f"No base_score JSONs found in {base_scores_dir}"
    records = [json.load(open(f)) for f in files]

    base_probs  = np.array([r["prediction"]["softmax"] for r in records], dtype=np.float32)
    base_scores = np.array([r["prediction"]["base_score"] for r in records], dtype=np.float32)
    pred_ef     = np.array([r["echo"]["predicted_ef"] for r in records], dtype=np.float32)
    gt_ef       = np.array([r["echo"]["ground_truth_ef"] for r in records], dtype=np.float32)
    patient_ids = [r["patient_id"] for r in records]

    def ef_to_class(ef):
        if ef >= 50: return 0
        elif ef >= 40: return 1
        else: return 2

    base_true = np.array([ef_to_class(e) for e in gt_ef], dtype=np.int64)
    base_pred = base_probs.argmax(axis=1)

    return {
        "patient_ids": patient_ids,
        "base_probs": base_probs,
        "base_scores": base_scores,
        "pred_ef": pred_ef,
        "gt_ef": gt_ef,
        "true_labels": base_true,
        "pred_labels": base_pred,
    }


# ─────────────────────────────────────────────
# TUSHAR (Intern 3) — Extended Module
# ─────────────────────────────────────────────

def reconstruct_extended_3class(common_alpha: np.ndarray,
                                 ef: np.ndarray,
                                 sigmoid_width: float = 3.0) -> np.ndarray:
    """
    Fix for the Mild=0 collapse. Builds a proper 3-class probability
    vector from common_alpha (disease probability) and ef_prediction
    (used only to split disease into Mild vs Severe).
    """
    p_disease = common_alpha
    p_normal = 1.0 - p_disease

    # Sigmoid split centered at EF=40: low EF -> more Severe weight
    p_severe_given_disease = 1.0 / (1.0 + np.exp((ef - 40.0) / sigmoid_width))
    p_mild_given_disease = 1.0 - p_severe_given_disease

    p_mild = p_disease * p_mild_given_disease
    p_severe = p_disease * p_severe_given_disease

    probs = np.stack([p_normal, p_mild, p_severe], axis=1)
    probs = np.clip(probs, 1e-6, None)
    return (probs / probs.sum(axis=1, keepdims=True)).astype(np.float32)


def load_tushar_data(embeddings_csv: Path,
                     predictions_json: Path,
                     features_parquet: Path,
                     clinical_parquet: Path,
                     calibrator_pkl: Path) -> dict:
    # -- embeddings (512-d) --
    df_emb = pd.read_csv(embeddings_csv)
    emb_sorted = df_emb.sort_values("Patient_ID", key=lambda x: x.map(pat_num)).reset_index(drop=True)
    feat_cols = [c for c in df_emb.columns if c.startswith("Feature_")]
    ext_emb = emb_sorted[feat_cols].values.astype(np.float32)

    # -- raw predictions (broken mild column, kept for reference/diagnostics) --
    with open(predictions_json) as f:
        preds_raw = json.load(f)
    pred_sorted = sorted(preds_raw, key=lambda d: pat_num(d["patient_id"]))

    ext_probs_raw = np.array(
        [[d["class_predictions"]["normal"], d["class_predictions"]["mild"],
          d["class_predictions"]["severe"]] for d in pred_sorted],
        dtype=np.float32,
    )
    ext_risk_score = np.array([d["risk_score"] for d in pred_sorted], dtype=np.float32)
    ext_ef = np.array([d["ef_prediction"] for d in pred_sorted], dtype=np.float32)

    # -- features + true labels --
    df_feat = pd.read_parquet(features_parquet)
    df_test = df_feat[df_feat["split"] == "test"].sort_values(
        "patient_id", key=lambda x: x.map(pat_num)).reset_index(drop=True)
    true_labels_binary = df_test["target"].values.astype(np.int64)
    original_patient_ids = df_test["patient_id"].tolist()

    # -- clinical scoring (Dempster-Shafer fused score) --
    df_clin = pd.read_parquet(clinical_parquet)
    df_clin_test = df_clin[df_clin["patient_id"].isin(df_test["patient_id"])].sort_values(
        "patient_id", key=lambda x: x.map(pat_num)).reset_index(drop=True)
    common_alpha = df_clin_test["common_alpha"].values.astype(np.float32)

    # -- FIXED 3-class probs (this is what should be used everywhere downstream) --
    ext_probs = reconstruct_extended_3class(common_alpha, ext_ef)

    # -- calibrator --
    with open(calibrator_pkl, "rb") as f:
        tushar_calibrator = pickle.load(f)

    assert len(emb_sorted) == len(pred_sorted) == len(df_test) == len(df_clin_test), \
        "Alignment error: counts don't match across Tushar's files!"

    return {
        "patient_ids": original_patient_ids,
        "ext_emb": ext_emb,
        "ext_probs": ext_probs,              # FIXED — use this
        "ext_probs_raw": ext_probs_raw,       # broken original, kept for diagnostics only
        "ext_risk_score": ext_risk_score,
        "ext_ef": ext_ef,
        "common_alpha": common_alpha,
        "true_labels_bin": true_labels_binary,
        "tushar_calibrator": tushar_calibrator,
    }


# ─────────────────────────────────────────────
# Simulated Base probs for Tushar's population
# (used until Vasanth shares ecg_embeddings.pt for true intermediate fusion)
# ─────────────────────────────────────────────

def ef_to_probability(ef: float) -> np.ndarray:
    centers = np.array([60.0, 45.0, 25.0])
    sigma = 8.0
    scores = np.exp(-((ef - centers) ** 2) / (2 * sigma ** 2))
    return (scores / scores.sum()).astype(np.float32)


def build_simulated_base_probs(ext_ef: np.ndarray) -> np.ndarray:
    return np.array([ef_to_probability(ef) for ef in ext_ef])


# ─────────────────────────────────────────────
# Validation checks (Task 4.1 requirement)
# ─────────────────────────────────────────────

def validate_inputs(base_probs: np.ndarray, ext_probs: np.ndarray) -> dict:
    """Run the three required validation checks from Task 4.1."""
    report = {}

    base_sum_ok = np.allclose(base_probs.sum(axis=1), 1.0, atol=1e-3)
    ext_sum_ok = np.allclose(ext_probs.sum(axis=1), 1.0, atol=1e-3)
    report["probabilities_sum_to_1"] = bool(base_sum_ok and ext_sum_ok)

    report["same_sample_count"] = (len(base_probs) == len(ext_probs))
    report["same_patient_ordering"] = "N/A — disjoint patient populations (see README)"

    report["base_n_samples"] = int(len(base_probs))
    report["ext_n_samples"] = int(len(ext_probs))

    return report


if __name__ == "__main__":
    # quick smoke test
    print("data_loader.py — run via run_fusion.py, not standalone.")
