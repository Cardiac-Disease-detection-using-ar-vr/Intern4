# Intern 4 — Unified Multi-Modal Fusion (Phase II)

Aditi Panda — Central integration point for the Cardiac VR project.

## What this module does

Combines:
- **Base Model** (Intern 2 / Vasanth + Amrutha): ECG + Echo fusion outputs
- **Extended Model** (Intern 3 / Tushar): Biosensor + Lipid risk outputs

into a single calibrated cardiac risk prediction (`unified_score.json`),
handed off to Intern 5 (Prasanna) for VR rendering.

## Critical data issue — read this first

Vasanth's and Tushar's models were trained on **disjoint patient
populations** (EchoNet-Dynamic vs. a separate clinical/wearable cohort)
with no overlapping patient IDs. This breaks the spec's assumed
"same patient ordering across inputs" precondition.

**Resolution:** all fusion is performed at score level on Tushar's
1,026-patient test cohort. Tushar's own `ef_prediction` field is used
to simulate what the Base module's output would look like for the same
patients, via the same Gaussian EF→probability mapping Vasanth's notebook
uses internally. This is documented as a limitation, not hidden.

**Action item:** once Vasanth shares `ecg_embeddings.pt` (paired,
real ECG/Echo embeddings for the same patient cohort), replace
`build_simulated_base_probs()` in `data_loader.py` with the real
embeddings for true intermediate/cross-attention fusion.

## Second critical issue — Tushar's "mild" probability was broken

`predictions_v6.json`'s `class_predictions.mild` field is capped near
0.15 across all 1,026 patients and **never wins argmax**. His underlying
model is effectively binary (healthy vs. disease); "mild" was never a
trained, usable output.

**Fix applied** (`data_loader.reconstruct_extended_3class`): rebuilds a
proper 3-class distribution from two signals that ARE meaningful:
- `common_alpha` (Dempster-Shafer disease probability, from
  `clinical_scoring_results_v6.parquet`)
- `ef_prediction` (continuous EF estimate, used only to split disease
  probability into Mild vs. Severe via a sigmoid centered at EF=40)

This brought Mild predictions from 0 patients to ~72, matching the
EF-based ground-truth Mild count of 73.

## Fusion strategies implemented (Task 4.2 / 4.3)

| Strategy | File | Notes |
|---|---|---|
| Early Fusion | `fusion_models.py::EarlyFusionModel` | Concatenates raw 3-d simulated base probs + 512-d Tushar embeddings, single classifier |
| Intermediate Fusion | `fusion_models.py::IntermediateFusionModel` | Separate projection heads per modality before fusing |
| Late Fusion | `fusion_models.py::late_fusion` | Weighted average of probability vectors, alpha swept 0.0-1.0 |
| Cross-Attention (primary) | `fusion_models.py::CrossAttentionFusion` | 2-token multi-head attention, outputs 768-d unified latent as required |

## Meta-learner (Task 4.4)

`meta_learner.py` stacks probability outputs from **all 5** models
(Base, Extended, Early, Intermediate, Cross-Attention = 15 features)
into an XGBoost classifier, trained with 5-fold stratified
cross-validation. Out-of-fold predictions are used for honest
evaluation (not in-sample).

## Calibration (Task 4.5)

Isotonic regression is fit per-class on the validation split. ECE and
MCE are computed before/after. Reliability diagrams (overall +
per-class) are generated to `plots/`.

**Important caveat documented in the code:** if isotonic calibration
is found to regress balanced accuracy by more than 3 percentage points
(this happened with Intermediate Fusion on this run: 0.96 → 0.82), the
pipeline exports the **calibrated probabilities** for confidence/ECE
reporting, but derives `fused_risk_score` (and therefore the implicit
class decision) from the **uncalibrated** probabilities. This protects
downstream accuracy while still reporting honest calibration metrics.
This guard is in `run_fusion.py`, controlled by
`prefer_uncalibrated_class`.

## Output schema (Task 4.6)

Exactly matches the spec:

```json
{
  "patient_id": "PAT_00003",
  "timestamp": "2026-06-30T04:51:02Z",
  "final_predictions": {"normal": 0.19, "mild": 0.35, "severe": 0.45},
  "calibrated_probabilities": {"normal": 0.51, "mild": 0.13, "severe": 0.36},
  "fused_risk_score": 81,
  "meta_learner_weights": [1.0],
  "calibration_params": {...},
  "model_ensemble_version": "v1.0_Intermediate_Fusion"
}
```

## How to run

```bash
pip install torch xgboost scikit-learn pandas numpy reportlab matplotlib --break-system-packages

# Place upstream data under data/:
#   data/INTERN-2-main/interim/base_scores/*.json
#   data/extended_embeddings.csv
#   data/Cardio_Tushar-main/{predictions_v6.json, features.parquet,
#                            clinical_scoring_results_v6.parquet,
#                            isotonic_calibration_frozen.pkl}

python run_fusion.py
```

## Deliverables produced

| File | Location |
|---|---|
| `unified_fusion.pt` | `models/` — Cross-Attention model state dict |
| `unified_score.json` × 1,026 | `interim/unified_scores/` |
| `fusion_comparison_report.pdf` | `reports/` |
| `calibration_plots.png` + reliability diagram | `plots/` |
| `xgboost_stacker.pkl` | `models/` |
| `README_UNIFIED.md` | this file |

## Results summary (this run)

| Model | Bal.Acc | F1 | AUROC | ECE | MCE |
|---|---|---|---|---|---|
| Base alone (Vasanth) | 0.547 | 0.563 | 0.818 | 0.264 | 0.350 |
| Extended alone (Tushar, fixed) | 0.834 | 0.857 | 0.987 | 0.037 | 0.320 |
| Early Fusion | 0.615 | 0.385 | 0.654 | 0.375 | 0.375 |
| **Intermediate Fusion** | **0.960** | **0.916** | 0.973 | 0.379 | 0.469 |
| Late Fusion (α=0.1) | 0.944 | 0.935 | **0.992** | 0.043 | 0.605 |
| Cross-Attention Fusion | 0.939 | 0.913 | 0.967 | 0.198 | 0.506 |
| XGBoost Meta-Learner (5-fold) | 0.875 | 0.875 | 0.982 | 0.021 | 0.262 |

**Best for accuracy:** Intermediate Fusion (0.960 Bal.Acc)
**Best for ranking:** Late Fusion α=0.1 (0.992 AUROC)
**Best for calibration:** XGBoost Meta-Learner (0.021 ECE)

No single strategy dominates on every axis — this is expected and is
documented honestly in `fusion_comparison_report.pdf`. Production
recommendation: use Intermediate Fusion's class decision with
XGBoost's confidence calibration once Vasanth's real embeddings are
available to retrain Intermediate/Cross-Attention on true paired data.
