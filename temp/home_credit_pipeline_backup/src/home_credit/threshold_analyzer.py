import os
from typing import Optional

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import precision_score, recall_score, f1_score

from .utils.logger import get_logger


class ThresholdAnalyzer:
    """Sweep probability thresholds and (optionally) plot precision/recall/F1 trade-offs."""

    def __init__(
        self,
        output_dir: str = "artifacts",
        step: float = 0.05,
        filename: str = "threshold_sweep.png",
        verbose: bool = True,
    ):
        self.output_dir = output_dir
        self.step = step
        self.filename = filename
        self.verbose = verbose
        self.logger = get_logger(self.__class__.__name__)
        os.makedirs(self.output_dir, exist_ok=True)

    def run(self, y_true, y_proba, plot: bool = True) -> float:
        y_true = np.asarray(y_true).astype(int)
        y_proba = np.asarray(y_proba).astype(float)

        thresholds = np.arange(0.05, 0.95, self.step)
        precisions: list[float] = []
        recalls: list[float] = []
        f1s: list[float] = []

        for thr in thresholds:
            y_pred = (y_proba >= thr).astype(int)
            precisions.append(float(precision_score(y_true, y_pred, zero_division=0)))
            recalls.append(float(recall_score(y_true, y_pred, zero_division=0)))
            f1s.append(float(f1_score(y_true, y_pred, zero_division=0)))

        best_idx = int(np.argmax(f1s))
        best_thr = float(thresholds[best_idx])

        if plot:
            plt.figure(figsize=(7, 5))
            sns.lineplot(x=thresholds, y=precisions, label="Precision")
            sns.lineplot(x=thresholds, y=recalls, label="Recall")
            sns.lineplot(x=thresholds, y=f1s, label="F1")
            plt.axvline(best_thr, linestyle="--", label=f"Best F1 thr={best_thr:.2f}")
            plt.xlabel("Threshold")
            plt.ylabel("Score")
            plt.title("Threshold Sweep")
            plt.legend()

            path = os.path.join(self.output_dir, self.filename)
            plt.tight_layout()
            plt.savefig(path, dpi=200)
            plt.close()

            if self.verbose:
                self.logger.info(f"Saved threshold sweep plot: {path}")

        if self.verbose:
            self.logger.info(f"Best F1 threshold: {best_thr:.3f} (F1={f1s[best_idx]:.3f})")

        return best_thr
