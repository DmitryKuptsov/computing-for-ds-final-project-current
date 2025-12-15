from __future__ import annotations

from typing import Optional

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from home_credit.utils.logger import get_logger


class Preprocessor:
    """Builds a ColumnTransformer for numeric/binary/categorical features."""

    def __init__(
        self,
        impute_strategy: str = "median",
        use_scaler: bool = False,
        verbose: bool = False,
    ):
        """
        Parameters
        ----------
        impute_strategy:
            Strategy for continuous numeric imputation (median/mean/most_frequent).
        use_scaler:
            Whether to scale continuous numeric features. For tree models (LightGBM)
            scaling is usually unnecessary, so default is False.
        verbose:
            If True, logs detected feature groups.
        """
        self.impute_strategy = impute_strategy
        self.use_scaler = use_scaler
        self.verbose = verbose
        self.logger = get_logger(self.__class__.__name__)
        self.transformer: Optional[ColumnTransformer] = None

    @staticmethod
    def _make_onehot() -> OneHotEncoder:
        """
        Create OneHotEncoder with compatibility across sklearn versions:
        - newer sklearn uses sparse_output
        - older sklearn uses sparse
        """
        try:
            return OneHotEncoder(handle_unknown="ignore", sparse_output=True)
        except TypeError:
            return OneHotEncoder(handle_unknown="ignore", sparse=True)

    def build(self, X: pd.DataFrame) -> ColumnTransformer:
        """Build (but do not fit) the preprocessing transformer."""
        numeric_cols = X.select_dtypes(include=["number", "bool"]).columns.tolist()
        categorical_cols = X.select_dtypes(include=["object", "category"]).columns.tolist()

        # binary numeric columns: values subset of {0,1} (ignoring NaNs)
        binary_cols = [
            col for col in numeric_cols
            if set(pd.Series(X[col]).dropna().unique()).issubset({0, 1})
        ]
        continuous_cols = [col for col in numeric_cols if col not in binary_cols]

        num_steps = [("imputer", SimpleImputer(strategy=self.impute_strategy))]
        if self.use_scaler:
            # with_mean=False keeps it compatible with sparse outputs
            num_steps.append(("scaler", StandardScaler(with_mean=False)))
        num_pipe = Pipeline(steps=num_steps)

        bin_pipe = Pipeline(
            steps=[("imputer", SimpleImputer(strategy="most_frequent"))]
        )

        cat_pipe = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("encoder", self._make_onehot()),
            ]
        )

        self.transformer = ColumnTransformer(
            transformers=[
                ("num", num_pipe, continuous_cols),
                ("bin", bin_pipe, binary_cols),
                ("cat", cat_pipe, categorical_cols),
            ],
            remainder="drop",
        )

        if self.verbose:
            self.logger.info(
                f"Columns detected: continuous={len(continuous_cols)}, "
                f"binary={len(binary_cols)}, categorical={len(categorical_cols)}"
            )

        return self.transformer
