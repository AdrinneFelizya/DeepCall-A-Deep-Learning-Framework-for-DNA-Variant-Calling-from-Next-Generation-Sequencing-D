"""
Evaluation Metrics for Variant Calling.

Includes the DL.zip metrics (accuracy, precision, recall, F1, confusion matrix)
plus site-level variant-calling metrics for real-data evaluation against truth VCFs.
"""
from __future__ import annotations

import numpy as np
import torch
from sklearn.metrics import (
    precision_recall_fscore_support, confusion_matrix, accuracy_score
)


def compute_metrics(all_labels, all_predictions, num_classes=4):
    """Compute precision, recall, F1-score, and accuracy (per-class)."""
    precision, recall, f1, _ = precision_recall_fscore_support(
        all_labels, all_predictions, average=None, labels=range(num_classes)
    )
    accuracy = accuracy_score(all_labels, all_predictions)
    cm = confusion_matrix(all_labels, all_predictions, labels=range(num_classes))

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "confusion_matrix": cm,
    }


def plot_confusion_matrix(cm, class_names, save_path=None):
    """Plot and optionally save a confusion matrix."""
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    ax.figure.colorbar(im, ax=ax)

    ax.set(
        xticks=np.arange(cm.shape[1]),
        yticks=np.arange(cm.shape[0]),
        xticklabels=class_names,
        yticklabels=class_names,
        ylabel="True label",
        xlabel="Predicted label",
        title="Confusion Matrix",
    )
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    thresh = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(
                j, i, format(cm[i, j], "d"),
                ha="center", va="center",
                color="white" if cm[i, j] > thresh else "black",
            )

    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig


def site_level_summary(precision, recall, f1, tp, fp, fn):
    """Return a formatted string of site-level metrics (for real-data paper tables)."""
    return (
        f"TP={tp}  FP={fp}  FN={fn}  "
        f"Precision={precision:.4f}  Recall={recall:.4f}  F1={f1:.4f}"
    )


__all__ = ["compute_metrics", "plot_confusion_matrix", "site_level_summary"]