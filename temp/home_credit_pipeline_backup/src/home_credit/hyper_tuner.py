import logging
from typing import Any

import numpy as np
import optuna
import pandas as pd

from .model_trainer import ModelTrainer
from .utils.logger import get_logger


class HyperTuner:
    """Optuna tuning for LightGBM using leakage-safe CV from ModelTrainer."""

    def __init__(
        self,
        n_trials: int = 30,
        n_splits: int = 5,
        random_state: int = 42,
    ):
        self.n_trials = n_trials
        self.n_splits = n_splits
        self.random_state = random_state
        self.logger = get_logger(self.__class__.__name__)
        self.best_params_: dict[str, Any] | None = None
        self.best_value_: float | None = None

    def _suggest_params(self, trial: optuna.Trial) -> dict[str, Any]:
        """Define Optuna search space."""
        return {
            "n_estimators": trial.suggest_int("n_estimators", 300, 1200, step=100),
            "learning_rate": trial.suggest_float("learning_rate", 0.005, 0.05, log=True),
            "num_leaves": trial.suggest_int("num_leaves", 16, 128),
            "max_depth": trial.suggest_int("max_depth", 3, 12),
            "min_child_samples": trial.suggest_int("min_child_samples", 20, 200),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
        }

    def tune(
        self,
        X_df: pd.DataFrame,
        y: np.ndarray,
        base_params: dict[str, Any],
        impute_strategy: str = "median",
        balance_strategy: str = "none",
    ) -> dict[str, Any]:
        """
        Run Optuna optimization and return best tuned parameters (subset).
        Uses ModelTrainer.cross_validate_oof to avoid preprocessing leakage.
        """
        self.logger.info(
            f"Starting Optuna tuning ({self.n_trials} trials, {self.n_splits}-fold CV)"
        )

        sampler = optuna.samplers.TPESampler(seed=self.random_state)
        study = optuna.create_study(direction="maximize", sampler=sampler)

        y = np.asarray(y).astype(int)

        def objective(trial: optuna.Trial) -> float:
            trial_params = self._suggest_params(trial)

            params = dict(base_params)
            params.update(trial_params)
            # keep reproducibility / performance knobs if user set them
            params.setdefault("random_state", self.random_state)
            params.setdefault("n_jobs", -1)
            params.setdefault("verbosity", -1)

            trainer = ModelTrainer(
                params=params,
                model_path="artifacts/_tmp_model.joblib",
                transformer_path="artifacts/_tmp_preprocessor.joblib",
                n_splits=self.n_splits,
                random_state=self.random_state,
            )
            # reduce log noise during tuning
            trainer.logger.setLevel(logging.WARNING)

            mean_auc, _ = trainer.cross_validate_oof(
                X_df=X_df,
                y=y,
                impute_strategy=impute_strategy,
                balance_strategy=balance_strategy,
            )
            return float(mean_auc)

        study.optimize(objective, n_trials=self.n_trials)

        self.best_params_ = study.best_params
        self.best_value_ = float(study.best_value)

        self.logger.info(f"Best CV ROC-AUC: {self.best_value_:.4f}")
        self.logger.info(f"Best parameters: {self.best_params_}")

        return dict(self.best_params_)
