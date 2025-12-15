import json
import os
import time
from typing import Dict

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)

from .utils.logger import get_logger


class Evaluator:
    """Evaluate binary classifier probabilities, save metrics and confusion matrix."""

    def __init__(self, metrics_path: str, figures_dir: str = "artifacts", verbose: bool = True):
        self.metrics_path = metrics_path
        self.figures_dir = figures_dir
        self.verbose = verbose
        self.logger = get_logger(self.__class__.__name__)

    def _find_best_threshold(self, y_true: np.ndarray, y_proba: np.ndarray) -> float:
        """Return threshold that maximizes F1 on the precision-recall curve."""
        precision, recall, thresholds = precision_recall_curve(y_true, y_proba)

        # precision_recall_curve returns thresholds of length (len(precision)-1)
        # Align F1 computation to thresholds by dropping the last precision/recall point.
        precision_t = precision[:-1]
        recall_t = recall[:-1]

        f1_scores = 2 * precision_t * recall_t / (precision_t + recall_t + 1e-8)
        best_idx = int(np.nanargmax(f1_scores))
        return float(thresholds[best_idx])

    def _plot_confusion_matrix(self, y_true: np.ndarray, y_pred: np.ndarray, normalize: bool = True) -> str:
        """Plot confusion matrix and save to figures_dir. Returns saved path."""
        cm = confusion_matrix(y_true, y_pred)

        if normalize:
            cm = cm.astype(float)
            row_sums = cm.sum(axis=1, keepdims=True)
            row_sums[row_sums == 0] = 1.0  # avoid division by zero
            cm = cm / row_sums

        plt.figure(figsize=(6, 5))
        sns.heatmap(
            cm,
            annot=True,
            fmt=".2f" if normalize else "d",
            cmap="Blues",
            xticklabels=["No Default", "Default"],
            yticklabels=["No Default", "Default"],
        )
        plt.xlabel("Predicted")
        plt.ylabel("Actual")
        plt.title("Confusion Matrix" + (" (Normalized)" if normalize else ""))

        os.makedirs(self.figures_dir, exist_ok=True)
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        path = os.path.join(self.figures_dir, f"confusion_matrix_{timestamp}.png")

        plt.tight_layout()
        plt.savefig(path, dpi=200)
        plt.close()

        if self.verbose:
            self.logger.info(f"Saved confusion matrix: {path}")

        return path

    def evaluate(self, y_true: np.ndarray, y_proba: np.ndarray) -> Dict[str, float]:
        """Compute metrics using the best F1 threshold, save JSON + confusion matrix."""
        y_true = np.asarray(y_true).astype(int)
        y_proba = np.asarray(y_proba).astype(float)

        best_threshold = self._find_best_threshold(y_true, y_proba)
        y_pred = (y_proba >= best_threshold).astype(int)

        metrics: Dict[str, float] = {
            "ROC_AUC": float(roc_auc_score(y_true, y_proba)),
            "PR_AUC": float(average_precision_score(y_true, y_proba)),
            "Balanced_Accuracy": float(balanced_accuracy_score(y_true, y_pred)),
            "F1": float(f1_score(y_true, y_pred, zero_division=0)),
            "Precision": float(precision_score(y_true, y_pred, zero_division=0)),
            "Recall": float(recall_score(y_true, y_pred, zero_division=0)),
            "Accuracy": float(accuracy_score(y_true, y_pred)),
            "Best_Threshold": float(best_threshold),
        }

        os.makedirs(os.path.dirname(self.metrics_path), exist_ok=True)
        with open(self.metrics_path, "w") as f:
            json.dump(metrics, f, indent=4)

        if self.verbose:
            self.logger.info(f"Saved metrics: {self.metrics_path}")

        self._plot_confusion_matrix(y_true, y_pred, normalize=True)

        return metrics
