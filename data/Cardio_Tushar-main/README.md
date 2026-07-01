# Cardiovascular Risk Scoring — Dempster-Shafer Fusion Pipeline

**Author:** Tushar  
**Model Version:** v6_final  
**Architecture:** Constrained Elastic Net Logistic Scorer → Isotonic Calibration → Murphy-Dempster Evidence Fusion

---

## Key Metrics

| Metric | Value |
|--------|-------|
| Cross-validated AUC | **0.9795** |
| Fused AUC (common_alpha) | **0.9763** |
| Accuracy | **96.20%** |
| Cohen's Kappa | 0.7605 |
| ECE (calibration error) | 0.0041 |
| Stability Index | 0.9958 |
| Dempster-rule usage | 100% (6837/6837) |

## Alpha Scores

| Score | Value |
|-------|-------|
| Mean α₁ (Clinical Score) | 0.0998 |
| α₂ (Model Performance) | 0.5019 (frozen scalar) |
| Mean Fused (common_alpha) | 0.1001 |

## Deep Learning Model

**HybridModel** — TCN + FT-Transformer Fusion
- **TCN Encoder:** 4 residual blocks (2→64→128→256→256) with BiGRU
- **Tab Encoder:** FT-Transformer (64-dim, 2 attention layers)
- **Fusion:** Gated MLP → 320-dim embedding
- **Heads:** Binary classifier (2-class) + Score head (risk regression)
- **Parameters:** 1,222,219

## File Manifest

| File | Description |
|------|-------------|
| `clinical_scoring_complete_dataset_v6.json` | Source of truth — full patient dataset (6,837 patients) |
| `clinical_scoring_results_v6.json` | Per-patient scoring results (α₁, α₂, fused scores) |
| `clinical_scoring_results_v6.parquet` | Same results in Parquet format |
| `predictions_v6.json` | Model predictions (class probs, risk score, EF) |
| `extended_score_v6.json` | Extended scoring breakdown per patient |
| `model_config_v6.json` | Model configuration and hyperparameters |
| `model.pt` | Trained PyTorch TCN+Transformer model weights |
| `features.parquet` | Feature matrix (6,837 × 33) |
| `isotonic_calibration_frozen.pkl` | Frozen isotonic calibration for VR runtime |
| `performance_analysis_v6.png` | Performance visualization chart |
| `complete_validation.py` | Validation suite (45/45 tests pass) |

## Validation

Run the validation suite to verify all formulas and data integrity:

```bash
python complete_validation.py
```

**Expected output:** 45/45 tests PASS across 8 validation parts:
- Part A: Alpha-1 Formula (7 tests)
- Part B: Alpha-2 Formula (4 tests)
- Part C: Common-Alpha Fusion (9 tests)
- Part D: 12-Method Validation Suite (12 tests)
- Part E: Metric Recomputation (3 tests)
- Part F: Correlations (3 tests)
- Part G: Cross-Validation Folds (2 tests)
- Part H: Data Integrity (5 tests)

## Pipeline Overview

```
Raw Features (31) → Elastic Net Logistic Regression → α₁ (clinical score)
                  → 10-Fold Stratified CV OOF metrics → α₂ (model performance scalar)
                  → Dempster-Shafer BPA Fusion → common_alpha (fused score)
                  → Isotonic Calibration → Calibrated Risk Probability
```

## VR Integration

The `isotonic_calibration_frozen.pkl` can be loaded directly in the VR runtime:

```python
import pickle
with open('isotonic_calibration_frozen.pkl', 'rb') as f:
    calibrator = pickle.load(f)

calibrated_risk = calibrator.predict([raw_alpha_common])[0]
```
Made by Tushar
