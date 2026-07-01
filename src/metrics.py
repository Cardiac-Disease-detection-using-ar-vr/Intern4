"""
src/intern4_fusion/metrics.py
================================
All evaluation metrics required by Task 4.5, plus the standard
classification metrics used to compare fusion strategies.
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.metrics import balanced_accuracy_score, f1_score, roc_auc_score

SEVERITY_LABELS = ["Normal", "Mild", "Severe"]


def expected_calibration_error(probs: np.ndarray, labels: np.ndarray,
                               n_bins: int = 10) -> float:
    """ECE — lower is better. Target < 0.05."""
    confidences = probs.max(axis=1)
    predictions = probs.argmax(axis=1)
    accuracies = (predictions == labels).astype(float)
    bin_bounds = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    for lo, hi in zip(bin_bounds[:-1], bin_bounds[1:]):
        mask = (confidences > lo) & (confidences <= hi)
        if mask.sum() > 0:
            avg_conf = confidences[mask].mean()
            avg_acc = accuracies[mask].mean()
            ece += (mask.sum() / len(probs)) * abs(avg_conf - avg_acc)
    return float(ece)


def maximum_calibration_error(probs: np.ndarray, labels: np.ndarray,
                              n_bins: int = 10) -> float:
    """MCE — the worst-case bin gap. Required by Task 4.5."""
    confidences = probs.max(axis=1)
    predictions = probs.argmax(axis=1)
    accuracies = (predictions == labels).astype(float)
    bin_bounds = np.linspace(0, 1, n_bins + 1)
    gaps = []
    for lo, hi in zip(bin_bounds[:-1], bin_bounds[1:]):
        mask = (confidences > lo) & (confidences <= hi)
        if mask.sum() > 0:
            avg_conf = confidences[mask].mean()
            avg_acc = accuracies[mask].mean()
            gaps.append(abs(avg_conf - avg_acc))
    return float(max(gaps)) if gaps else 0.0


def compute_metrics(probs: np.ndarray, true_labels: np.ndarray,
                    label_type: str = "binary") -> dict:
    """Balanced accuracy, macro F1, AUROC, ECE, MCE."""
    pred = probs.argmax(axis=1)

    if label_type == "binary":
        pred_bin = (pred > 0).astype(int)
        score_bin = probs[:, 1] + probs[:, 2]
        ba = balanced_accuracy_score(true_labels, pred_bin)
        f1 = f1_score(true_labels, pred_bin, average="macro", zero_division=0)
        auc = roc_auc_score(true_labels, score_bin)
        eval_labels = (pred > 0).astype(int)
    else:
        ba = balanced_accuracy_score(true_labels, pred)
        f1 = f1_score(true_labels, pred, average="macro", zero_division=0)
        try:
            auc = roc_auc_score(true_labels, probs, multi_class="ovr", average="macro")
        except Exception:
            auc = float("nan")
        eval_labels = true_labels

    ece = expected_calibration_error(probs, eval_labels)
    mce = maximum_calibration_error(probs, eval_labels)

    return {
        "balanced_acc": round(float(ba), 4),
        "macro_f1": round(float(f1), 4),
        "auroc": round(float(auc), 4),
        "ece": round(float(ece), 4),
        "mce": round(float(mce), 4),
    }


# ═══════════════════════════════════════════════════════════
# Reliability diagrams (Task 4.5 requirement)
# ═══════════════════════════════════════════════════════════

def plot_reliability_diagram(probs: np.ndarray, true_labels: np.ndarray,
                             out_path: Path, n_bins: int = 10,
                             title: str = "Reliability Diagram",
                             per_class: bool = True,
                             label_type: str = "binary"):
    """
    Generates a reliability diagram (confidence vs accuracy), with
    one subplot per class as required by Task 4.5.
    """
    if label_type == "binary":
        eval_labels = (true_labels > 0).astype(int) if true_labels.max() <= 2 else true_labels
    else:
        eval_labels = true_labels

    n_classes = probs.shape[1]
    fig, axes = plt.subplots(1, n_classes + 1, figsize=(5 * (n_classes + 1), 4.5))

    bin_bounds = np.linspace(0, 1, n_bins + 1)
    bin_centers = (bin_bounds[:-1] + bin_bounds[1:]) / 2

    # --- Overall (max-confidence) reliability ---
    ax = axes[0]
    confidences = probs.max(axis=1)
    predictions = probs.argmax(axis=1)
    overall_true = (eval_labels > 0).astype(int) if label_type == "binary" else eval_labels
    overall_pred = (predictions > 0).astype(int) if label_type == "binary" else predictions
    accuracies = (overall_pred == overall_true).astype(float)

    bin_acc, bin_conf = [], []
    for lo, hi in zip(bin_bounds[:-1], bin_bounds[1:]):
        mask = (confidences > lo) & (confidences <= hi)
        if mask.sum() > 0:
            bin_acc.append(accuracies[mask].mean())
            bin_conf.append(confidences[mask].mean())
        else:
            bin_acc.append(np.nan)
            bin_conf.append(np.nan)

    ax.plot([0, 1], [0, 1], "k--", alpha=0.5, label="Perfect calibration")
    ax.bar(bin_centers, bin_acc, width=0.08, alpha=0.7, color="steelblue",
           edgecolor="black", label="Accuracy")
    ax.set_xlabel("Confidence")
    ax.set_ylabel("Accuracy")
    ax.set_title("Overall (max confidence)")
    ax.legend(fontsize=8)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)

    # --- Per-class reliability ---
    for c in range(n_classes):
        ax = axes[c + 1]
        class_probs = probs[:, c]
        class_true = (eval_labels == c).astype(float)

        bin_acc_c, bin_conf_c = [], []
        for lo, hi in zip(bin_bounds[:-1], bin_bounds[1:]):
            mask = (class_probs > lo) & (class_probs <= hi)
            if mask.sum() > 0:
                bin_acc_c.append(class_true[mask].mean())
                bin_conf_c.append(class_probs[mask].mean())
            else:
                bin_acc_c.append(np.nan)
                bin_conf_c.append(np.nan)

        ax.plot([0, 1], [0, 1], "k--", alpha=0.5)
        ax.bar(bin_centers, bin_acc_c, width=0.08, alpha=0.7,
               color=["seagreen", "goldenrod", "firebrick"][c],
               edgecolor="black")
        ax.set_xlabel(f"Predicted P({SEVERITY_LABELS[c]})")
        ax.set_ylabel("Empirical frequency")
        ax.set_title(f"Class: {SEVERITY_LABELS[c]}")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)

    fig.suptitle(title, fontsize=13, fontweight="bold")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_before_after_calibration(probs_before: np.ndarray, probs_after: np.ndarray,
                                  true_labels: np.ndarray, out_path: Path,
                                  label_type: str = "binary"):
    """Side-by-side before/after calibration comparison (Task 4.5)."""
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))

    for ax, probs, title in zip(
        axes, [probs_before, probs_after], ["Before Calibration", "After Calibration"]
    ):
        confidences = probs.max(axis=1)
        predictions = probs.argmax(axis=1)
        if label_type == "binary":
            eval_true = (true_labels > 0).astype(int)
            eval_pred = (predictions > 0).astype(int)
        else:
            eval_true, eval_pred = true_labels, predictions
        accuracies = (eval_pred == eval_true).astype(float)

        bin_bounds = np.linspace(0, 1, 11)
        bin_centers = (bin_bounds[:-1] + bin_bounds[1:]) / 2
        bin_acc = []
        for lo, hi in zip(bin_bounds[:-1], bin_bounds[1:]):
            mask = (confidences > lo) & (confidences <= hi)
            bin_acc.append(accuracies[mask].mean() if mask.sum() > 0 else np.nan)

        ax.plot([0, 1], [0, 1], "k--", alpha=0.5, label="Perfect")
        ax.bar(bin_centers, bin_acc, width=0.08, color="steelblue",
               edgecolor="black", alpha=0.8)
        ece = expected_calibration_error(probs,
              (true_labels > 0).astype(int) if label_type == "binary" else true_labels)
        ax.set_title(f"{title}\nECE = {ece:.4f}")
        ax.set_xlabel("Confidence")
        ax.set_ylabel("Accuracy")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.legend(fontsize=8)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
