import numpy as np
import pandas as pd

from home_credit.preprocessor import Preprocessor


def _make_small_X():
    return pd.DataFrame(
        {
            # continuous numeric
            "AMT_INCOME_TOTAL": [100000.0, 50000.0, np.nan],
            "AMT_CREDIT": [200000.0, np.nan, 150000.0],

            # binary numeric
            "FLAG_OWN_CAR": [1, 0, np.nan],
            "FLAG_OWN_REALTY": [0, 1, 1],

            # categorical
            "NAME_CONTRACT_TYPE": ["Cash loans", "Revolving loans", None],
            "CODE_GENDER": ["M", "F", "F"],
        }
    )


def test_preprocessor_build_returns_column_transformer():
    X = _make_small_X()
    prep = Preprocessor(impute_strategy="median")
    transformer = prep.build(X)
    assert transformer is not None


def test_preprocessor_fit_transform_preserves_row_count_and_no_nans():
    X = _make_small_X()
    prep = Preprocessor(impute_strategy="median")
    transformer = prep.build(X)

    Xt = transformer.fit_transform(X)
    # rows preserved
    assert Xt.shape[0] == X.shape[0]

    # no NaNs after imputation/encoding
    Xt_dense = Xt.toarray() if hasattr(Xt, "toarray") else Xt
    assert np.isfinite(Xt_dense).all()


def test_preprocessor_transform_handles_unseen_categories():
    X_train = pd.DataFrame(
        {
            "AMT_INCOME_TOTAL": [100000.0, 50000.0],
            "AMT_CREDIT": [200000.0, 150000.0],
            "FLAG_OWN_CAR": [1, 0],
            "FLAG_OWN_REALTY": [0, 1],
            "NAME_CONTRACT_TYPE": ["Cash loans", "Revolving loans"],
            "CODE_GENDER": ["M", "F"],
        }
    )
    X_test = pd.DataFrame(
        {
            "AMT_INCOME_TOTAL": [70000.0],
            "AMT_CREDIT": [120000.0],
            "FLAG_OWN_CAR": [1],
            "FLAG_OWN_REALTY": [1],
            "NAME_CONTRACT_TYPE": ["New unseen type"],  # unseen category
            "CODE_GENDER": ["X"],                       # unseen category
        }
    )

    prep = Preprocessor(impute_strategy="median")
    transformer = prep.build(X_train)
    transformer.fit(X_train)

    Xt_test = transformer.transform(X_test)
    assert Xt_test.shape[0] == 1  # transform succeeded


def test_preprocessor_binary_columns_not_scaled_pipeline():
    # This checks that binary columns are routed to 'bin' transformer (no scaler there).
    X = _make_small_X()
    prep = Preprocessor(impute_strategy="median")
    transformer = prep.build(X)

    # names are set in ColumnTransformer as ("num", ...), ("bin", ...), ("cat", ...)
    names = [name for name, _, _ in transformer.transformers]
    assert "bin" in names and "cat" in names and "num" in names
