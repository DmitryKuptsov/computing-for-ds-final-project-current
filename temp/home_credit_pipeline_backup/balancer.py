import numpy as np
import pandas as pd
from typing import Tuple, Literal
from sklearn.utils import resample

try:
    from imblearn.over_sampling import SMOTE
except ImportError:
    SMOTE = None


class Balancer:
    """
    Handles class imbalance via oversampling, undersampling, or SMOTE.
    
    Example:
        balancer = Balancer(strategy="smote")
        Xb, yb = balancer.balance(X, y)
    """

    def __init__(
        self,
        strategy: Literal["none", "oversample", "undersample", "smote"] = "none",
        random_state: int = 42,
    ):
        self.strategy = strategy
        self.random_state = random_state

    def balance(self, X: np.ndarray | pd.DataFrame, y: np.ndarray | pd.Series) -> Tuple:
        if self.strategy == "none":
            return X, y

        print(f"Applying class balancing: {self.strategy}")

        y = np.asarray(y)
        X = np.asarray(X)

        n_pos = np.sum(y == 1)
        n_neg = np.sum(y == 0)

        if n_pos == 0 or n_neg == 0:
            print("Only one class present. Skipping balancing.")
            return X, y

        if self.strategy == "smote":
            if SMOTE is None:
                raise ImportError("Install imbalanced-learn to use SMOTE.")
            sm = SMOTE(random_state=self.random_state)
            X_res, y_res = sm.fit_resample(X, y)
            return X_res, y_res

        elif self.strategy == "oversample":
            X_pos = X[y == 1]
            X_neg = X[y == 0]
            n_to_add = n_neg - n_pos
            X_pos_up = resample(
                X_pos,
                replace=True,
                n_samples=n_to_add,
                random_state=self.random_state,
            )
            X_bal = np.vstack((X, X_pos_up))
            y_bal = np.hstack((y, np.ones(n_to_add)))
            return X_bal, y_bal

        elif self.strategy == "undersample":
            X_pos = X[y == 1]
            X_neg = X[y == 0]
            X_neg_down = resample(
                X_neg,
                replace=False,
                n_samples=len(X_pos),
                random_state=self.random_state,
            )
            X_bal = np.vstack((X_pos, X_neg_down))
            y_bal = np.hstack((np.ones(len(X_pos)), np.zeros(len(X_neg_down))))
            return X_bal, y_bal

        else:
            raise ValueError(f"Unknown balancing strategy: {self.strategy}")
