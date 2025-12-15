import os
import joblib
import warnings
from typing import Any

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

from .preprocessor import Preprocessor
from .utils.logger import get_logger


class ModelTrainer:
    """
    Trains LightGBM with leakage-safe cross-validation:
    preprocessing is fit only on training folds, then applied to validation folds.

    Provides:
      - cross_validate_oof: returns mean CV AUC and out-of-fold probabilities
      - fit_final: trains final model on full data and saves model + transformer
    """

    def __init__(
        self,
        params: dict[str, Any],
        model_path: str,
        transformer_path: str = "artifacts/preprocessor.joblib",
        n_splits: int = 5,
        random_state: int = 42,
    ):
        self.params = dict(params)
        self.model_path = model_path
        self.transformer_path = transformer_path
        self.n_splits = n_splits
        self.random_state = random_state

        self.logger = get_logger(self.__class__.__name__)
        self.models: list[LGBMClassifier] = []
        self.final_model: LGBMClassifier | None = None
        self.final_transformer = None

    @staticmethod
    def _compute_scale_pos_weight(y: np.ndarray) -> float:
        n_neg = np.sum(y == 0)
        n_pos = np.sum(y == 1)
        return float(max(1.0, n_neg / max(n_pos, 1)))

    def _balance_fold(
        self,
        X_train_df: pd.DataFrame,
        y_train: np.ndarray,
        strategy: str,
        fold_seed: int,
    ) -> tuple[pd.DataFrame, np.ndarray]:
        """
        Balance only within training fold, before preprocessing.
        Supports: none, oversample, undersample.
        SMOTE is intentionally not applied here to keep behavior stable with mixed types.
        """
        if strategy in (None, "none"):
            return X_train_df, y_train

        if strategy == "smote":
            # Keep as no-op for now; can be implemented later with a safe numeric pipeline.
            return X_train_df, y_train

        y_train = np.asarray(y_train)
        idx_pos = np.where(y_train == 1)[0]
        idx_neg = np.where(y_train == 0)[0]

        if len(idx_pos) == 0 or len(idx_neg) == 0:
            return X_train_df, y_train

        rng = np.random.RandomState(fold_seed)

        if strategy == "oversample":
            n_to_add = len(idx_neg) - len(idx_pos)
            if n_to_add <= 0:
                return X_train_df, y_train
            add_pos = rng.choice(idx_pos, size=n_to_add, replace=True)
            new_idx = np.concatenate([np.arange(len(y_train)), add_pos])
            Xb = X_train_df.iloc[new_idx].reset_index(drop=True)
            yb = y_train[new_idx]
            return Xb, yb

        if strategy == "undersample":
            keep_neg = rng.choice(idx_neg, size=len(idx_pos), replace=False)
            keep_idx = np.concatenate([idx_pos, keep_neg])
            rng.shuffle(keep_idx)
            Xb = X_train_df.iloc[keep_idx].reset_index(drop=True)
            yb = y_train[keep_idx]
            return Xb, yb

        raise ValueError(f"Unknown balancing strategy: {strategy}")

    def cross_validate_oof(
        self,
        X_df: pd.DataFrame,
        y: np.ndarray,
        impute_strategy: str = "median",
        balance_strategy: str = "none",
    ) -> tuple[float, np.ndarray]:
        """
        Stratified CV with fold-wise preprocessing. Returns:
          - mean ROC-AUC across folds
          - out-of-fold predicted probabilities for the positive class
        """
        warnings.filterwarnings("ignore", category=UserWarning, module="lightgbm")

        y = np.asarray(y).astype(int)
        oof_proba = np.zeros(len(y), dtype=float)
        fold_aucs: list[float] = []
        self.models = []

        skf = StratifiedKFold(
            n_splits=self.n_splits, shuffle=True, random_state=self.random_state
        )

        for fold, (train_idx, val_idx) in enumerate(skf.split(X_df, y), start=1):
            X_train_df = X_df.iloc[train_idx].reset_index(drop=True)
            y_train = y[train_idx]
            X_val_df = X_df.iloc[val_idx].reset_index(drop=True)
            y_val = y[val_idx]

            # Balance only training fold (optional)
            X_train_df, y_train = self._balance_fold(
                X_train_df,
                y_train,
                strategy=balance_strategy,
                fold_seed=self.random_state + fold,
            )

            # Fit preprocessing only on training fold (prevents leakage)
            prep = Preprocessor(impute_strategy=impute_strategy)
            transformer = prep.build(X_train_df)
            X_train = transformer.fit_transform(X_train_df)
            X_val = transformer.transform(X_val_df)

            params = dict(self.params)
            params["scale_pos_weight"] = self._compute_scale_pos_weight(y_train)
            params.setdefault("verbosity", -1)

            model = LGBMClassifier(**params)
            model.fit(X_train, y_train)

            val_proba = model.predict_proba(X_val)[:, 1]
            oof_proba[val_idx] = val_proba

            auc = roc_auc_score(y_val, val_proba)
            fold_aucs.append(float(auc))
            self.models.append(model)

            self.logger.info(f"Fold {fold}/{self.n_splits} ROC-AUC: {auc:.4f}")

        mean_auc = float(np.mean(fold_aucs))
        return mean_auc, oof_proba

    def fit_final(
        self,
        X_df: pd.DataFrame,
        y: np.ndarray,
        impute_strategy: str = "median",
    ) -> None:
        """
        Fit preprocessing + final model on the full dataset and save artifacts.
        """
        warnings.filterwarnings("ignore", category=UserWarning, module="lightgbm")

        y = np.asarray(y).astype(int)

        prep = Preprocessor(impute_strategy=impute_strategy)
        transformer = prep.build(X_df)
        X_full = transformer.fit_transform(X_df)

        params = dict(self.params)
        params["scale_pos_weight"] = self._compute_scale_pos_weight(y)
        params.setdefault("verbosity", -1)

        model = LGBMClassifier(**params)
        model.fit(X_full, y)

        self.final_model = model
        self.final_transformer = transformer

        os.makedirs(os.path.dirname(self.model_path), exist_ok=True)
        joblib.dump(model, self.model_path)

        os.makedirs(os.path.dirname(self.transformer_path), exist_ok=True)
        joblib.dump(transformer, self.transformer_path)

        self.logger.info(f"Saved model: {self.model_path}")
        self.logger.info(f"Saved transformer: {self.transformer_path}")

    def predict_proba_test(self, X_test_df: pd.DataFrame) -> np.ndarray:
        """
        Predict probabilities on a test dataframe using the saved final transformer/model.
        Assumes fit_final was called in the same process.
        """
        if self.final_model is None or self.final_transformer is None:
            raise RuntimeError("Call fit_final() before predict_proba_test().")

        X_test = self.final_transformer.transform(X_test_df)
        return self.final_model.predict_proba(X_test)[:, 1]
