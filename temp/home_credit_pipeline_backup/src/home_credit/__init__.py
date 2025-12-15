"""
Home Credit Default Risk — Modular Machine Learning Pipeline

This package provides an end-to-end implementation for
credit default risk prediction using LightGBM with
configurable preprocessing, feature engineering,
class balancing, hyperparameter tuning, and evaluation.

Modules:
    config              — Load YAML configuration safely.
    data_loader         — Read and optionally sample CSV data.
    feature_engineer    — Create domain-specific engineered features.
    preprocessor        — Impute, encode, and scale features.
    model_trainer       — Train LightGBM with cross-validation.
    evaluator           — Evaluate model and save confusion matrix.
    threshold_analyzer  — Analyze threshold trade-offs.
    hyper_tuner         — Tune hyperparameters with Optuna.
    pipeline            — Orchestrates all components.
    utils.logger        — Unified timestamped console logger.
"""

from .config import Config
from .data_loader import DataLoader
from .feature_engineer import FeatureEngineer
from .preprocessor import Preprocessor
# from .balancer import Balancer
from .model_trainer import ModelTrainer
from .evaluator import Evaluator
from .threshold_analyzer import ThresholdAnalyzer
from .hyper_tuner import HyperTuner
from .pipeline import PipelineRunner

__all__ = [
    "Config",
    "DataLoader",
    "FeatureEngineer",
    "Preprocessor",
    "ModelTrainer",
    "Evaluator",
    "ThresholdAnalyzer",
    "HyperTuner",
    "PipelineRunner",
]
