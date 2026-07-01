"""
src/intern4_fusion/meta_learner.py
=====================================
Task 4.4: XGBoost meta-learner stacked on ALL FIVE upstream signals:
  1. Base model probabilities         (3)
  2. Extended model probabilities     (3)
  3. Early fusion model probabilities (3)
  4. Intermediate fusion probabilities(3)
  5. Cross-attention fusion probabilities (3)
Total: 15 meta-features.

Uses 5-fold cross-validation as required.
"""

import numpy as np
import xgboost as xgb
from sklearn.model_selection import StratifiedKFold


def build_meta_feature_matrix(base_probs: np.ndarray,
                               ext_probs: np.ndarray,
                               early_probs: np.ndarray,
                               inter_probs: np.ndarray,
                               cross_attn_probs: np.ndarray) -> np.ndarray:
    """
    Concatenate probabilities from all 5 models into one meta-feature
    matrix, exactly as specified in Task 4.4.
    """
    return np.hstack([
        base_probs, ext_probs, early_probs, inter_probs, cross_attn_probs
    ]).astype(np.float32)   # (N, 15)


def train_xgboost_stacker_cv(X_meta: np.ndarray, y_binary: np.ndarray,
                              n_splits: int = 5, seed: int = 42) -> dict:
    """
    5-fold cross-validated XGBoost stacking, as required by Task 4.4.
    Returns the final model trained on all data plus out-of-fold (OOF)
    predictions for honest evaluation.
    """
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    oof_probs = np.zeros(len(y_binary))
    fold_models = []

    print(f"--- XGBoost Meta-Learner: {n_splits}-Fold Cross-Validation ---")
    for fold, (train_idx, val_idx) in enumerate(skf.split(X_meta, y_binary)):
        model = xgb.XGBClassifier(
            n_estimators=300,
            max_depth=4,
            learning_rate=0.05,
            objective="binary:logistic",
            eval_metric="logloss",
            tree_method="hist",
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=seed,
            verbosity=0,
        )
        model.fit(X_meta[train_idx], y_binary[train_idx])
        fold_probs = model.predict_proba(X_meta[val_idx])[:, 1]
        oof_probs[val_idx] = fold_probs
        fold_models.append(model)

        from sklearn.metrics import roc_auc_score, balanced_accuracy_score
        auc = roc_auc_score(y_binary[val_idx], fold_probs)
        ba = balanced_accuracy_score(y_binary[val_idx], (fold_probs > 0.5).astype(int))
        print(f"  Fold {fold + 1}/{n_splits} — AUROC={auc:.4f} BalAcc={ba:.4f}")

    # Final model trained on ALL data (for deployment)
    final_model = xgb.XGBClassifier(
        n_estimators=300, max_depth=4, learning_rate=0.05,
        objective="binary:logistic", eval_metric="logloss",
        tree_method="hist", subsample=0.8, colsample_bytree=0.8,
        random_state=seed, verbosity=0,
    )
    final_model.fit(X_meta, y_binary)

    from sklearn.metrics import roc_auc_score, balanced_accuracy_score, f1_score
    oof_auc = roc_auc_score(y_binary, oof_probs)
    oof_ba = balanced_accuracy_score(y_binary, (oof_probs > 0.5).astype(int))
    oof_f1 = f1_score(y_binary, (oof_probs > 0.5).astype(int), average="macro")

    print(f"\n  OOF (honest) performance: AUROC={oof_auc:.4f} "
          f"BalAcc={oof_ba:.4f} F1={oof_f1:.4f}")

    return {
        "final_model": final_model,
        "fold_models": fold_models,
        "oof_probs": oof_probs,
        "oof_auroc": round(float(oof_auc), 4),
        "oof_balanced_acc": round(float(oof_ba), 4),
        "oof_macro_f1": round(float(oof_f1), 4),
        "feature_importances": final_model.feature_importances_.tolist(),
    }


def meta_probs_to_3class(meta_prob_disease: np.ndarray,
                          ext_probs: np.ndarray) -> np.ndarray:
    """
    Convert the meta-learner's binary P(disease) output back into a
    3-class distribution using the Extended module's Mild/Severe split
    ratio (since the meta-learner itself is trained on binary ground truth).
    """
    disease_sum = ext_probs[:, 1] + ext_probs[:, 2] + 1e-9
    p_mild = meta_prob_disease * (ext_probs[:, 1] / disease_sum)
    p_severe = meta_prob_disease * (ext_probs[:, 2] / disease_sum)
    p_normal = 1 - meta_prob_disease
    return np.stack([p_normal, p_mild, p_severe], axis=1).astype(np.float32)
