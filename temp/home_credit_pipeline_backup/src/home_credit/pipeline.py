import os
import warnings
from textwrap import indent

import pandas as pd

from .config import Config
from .data_loader import DataLoader
from .evaluator import Evaluator
from .feature_engineer import FeatureEngineer
from .model_trainer import ModelTrainer
from .threshold_analyzer import ThresholdAnalyzer
from .utils.logger import get_logger
from .hyper_tuner import HyperTuner


class PipelineRunner:
    """End-to-end Home Credit Default Risk pipeline.

    Steps:
      1. Load and optionally sample training data
      2. Perform feature engineering
      3. Preprocess data (imputation, encoding, scaling)
      4. Optionally tune hyperparameters with Optuna
      5. Optionally balance classes (SMOTE, oversampling, undersampling)
      6. Train LightGBM model with stratified cross-validation
      7. Evaluate model (metrics, confusion matrix)
      8. Optionally analyze precision/recall trade-offs via threshold sweep
      9. Predict on test data and save Kaggle submission"""

    def __init__(self, config_path: str):
        self.config = Config.from_yaml(config_path)
        self.logger = get_logger(self.__class__.__name__)
        warnings.filterwarnings(
            "ignore",
            message="X does not have valid feature names",
            category=UserWarning,
            module="sklearn",
        )

    def run(self) -> None:
        cfg = self.config
        self.logger.info("Starting Home Credit training pipeline")

        df = DataLoader(cfg.data["path_train"], cfg.data.get("sample_size")).load()
        self.logger.info(f"Loaded dataset: {df.shape[0]:,} rows x {df.shape[1]} cols")

        df = FeatureEngineer().transform(df)

        target_col = cfg.data["target_col"]
        X_df = df.drop(columns=[target_col])
        y_full = df[target_col].astype(int).to_numpy()

        if cfg.model.get("tune", False):
            tuner = HyperTuner(
                n_trials=cfg.model.get("n_trials", 30),
                n_splits=cfg.validation.get("n_splits", 5),
                random_state=cfg.validation.get("random_state", 42),
            )
            best_params = tuner.tune(
                X_df=X_df,
                y=y_full,
                base_params=cfg.model["params"],
                impute_strategy=impute_strategy,
                balance_strategy=balance_strategy,
            )
            cfg.model["params"].update(best_params)
            self.logger.info("Model parameters updated with tuned values")
        else:
            self.logger.info("Hyperparameter tuning disabled")


        impute_strategy = cfg.preprocessing.get("impute_strategy", "median")
        balance_strategy = cfg.validation.get("balance_strategy", "none")

        trainer = ModelTrainer(
            params=cfg.model["params"],
            model_path=cfg.output["model_path"],
            transformer_path=cfg.output.get("transformer_path", "artifacts/preprocessor.joblib"),
            n_splits=cfg.validation.get("n_splits", 5),
            random_state=cfg.validation.get("random_state", 42),
        )

        mean_auc, oof_proba = trainer.cross_validate_oof(
            X_df=X_df,
            y=y_full,
            impute_strategy=impute_strategy,
            balance_strategy=balance_strategy,
        )
        self.logger.info(f"CV ROC-AUC (mean over folds): {mean_auc:.4f}")

        evaluator = Evaluator(cfg.output["metrics_path"], "artifacts")
        metrics = evaluator.evaluate(y_full, oof_proba)

        metrics_str = indent(
            "\n".join([f"{k}: {v:.4f}" for k, v in metrics.items() if isinstance(v, float)]),
            " " * 4,
        )
        self.logger.info(f"OOF metrics:\n{metrics_str}")

        if cfg.validation.get("analyze_thresholds", False):
            analyzer = ThresholdAnalyzer("artifacts")
            best_thr = analyzer.run(y_full, oof_proba)
            self.logger.info(f"Best F1 threshold (OOF): {best_thr:.3f}")

        # Fit final model on full data and save model + transformer
        trainer.fit_final(X_df=X_df, y=y_full, impute_strategy=impute_strategy)



        # Predict on test set (hardcoded path)
        test_path = "data/application_test.csv"
        submission_path = "artifacts/submission.csv"

        if not os.path.exists(test_path):
            self.logger.error("application_test.csv not found; skipping submission")
            return

        test_df = pd.read_csv(test_path)
        test_features = FeatureEngineer().transform(test_df)

        y_test_pred = trainer.predict_proba_test(test_features)

        submission = pd.DataFrame({"SK_ID_CURR": test_df["SK_ID_CURR"], "TARGET": y_test_pred})
        os.makedirs("artifacts", exist_ok=True)
        submission.to_csv(submission_path, index=False)

        self.logger.info(f"Submission saved: {submission_path} ({len(submission):,} rows)")
        self.logger.info(
            f"Test prediction stats (mean={y_test_pred.mean():.4f}, "
            f"min={y_test_pred.min():.4f}, max={y_test_pred.max():.4f})"
        )
        self.logger.info("Pipeline finished")
