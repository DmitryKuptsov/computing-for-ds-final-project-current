import numpy as np
import pandas as pd

from feature_engineer import FeatureEngineer


def test_feature_engineer_adds_expected_columns_basic():
    df = pd.DataFrame(
        {
            "SK_ID_CURR": [1, 2, 3],
            "AMT_INCOME_TOTAL": [100000.0, np.nan, 0.0],
            "AMT_CREDIT": [200000.0, 150000.0, 0.0],
            "AMT_ANNUITY": [20000.0, 10000.0, np.nan],
            "DAYS_EMPLOYED": [-1000, 0, -3650],
            "DAYS_BIRTH": [-12000, -20000, -10000],
        }
    )

    out = FeatureEngineer().transform(df)

    expected = {
        "INCOME_MISSING",
        "ANNUITY_MISSING",
        "DAYS_EMPLOYED_ANOM",
        "CREDIT_INCOME_RATIO",
        "ANNUITY_INCOME_RATIO",
        "AGE_YEARS",
        "EMPLOYMENT_YEARS",
        "EMPLOYED_TO_AGE",
    }
    assert expected.issubset(out.columns)


def test_feature_engineer_adds_expected_columns_ext_sources_and_goods_family():
    df = pd.DataFrame(
        {
            "SK_ID_CURR": [1, 2],
            "AMT_INCOME_TOTAL": [100000.0, 50000.0],
            "AMT_CREDIT": [200000.0, 150000.0],
            "AMT_ANNUITY": [20000.0, 10000.0],
            "AMT_GOODS_PRICE": [180000.0, 140000.0],
            "CNT_FAM_MEMBERS": [2.0, 4.0],
            "EXT_SOURCE_1": [0.5, np.nan],
            "EXT_SOURCE_2": [0.7, 0.2],
            "EXT_SOURCE_3": [0.9, 0.1],
            "DAYS_EMPLOYED": [-1000, FeatureEngineer.DAYS_EMPLOYED_ANOM],
            "DAYS_BIRTH": [-12000, -20000],
        }
    )

    out = FeatureEngineer().transform(df)

    expected = {
        "APPS_EXT_SOURCE_MEAN",
        "APPS_EXT_SOURCE_STD",
        "APPS_ANNUITY_CREDIT_RATIO",
        "APPS_GOODS_CREDIT_RATIO",
        "APPS_ANNUITY_INCOME_RATIO",
        "APPS_CREDIT_INCOME_RATIO",
        "APPS_GOODS_INCOME_RATIO",
        "APPS_CNT_FAM_INCOME_RATIO",
        "DAYS_EMPLOYED_ANOM",
    }
    assert expected.issubset(out.columns)

    # anomaly flag should mark the second row
    assert out.loc[0, "DAYS_EMPLOYED_ANOM"] == 0
    assert out.loc[1, "DAYS_EMPLOYED_ANOM"] == 1


def test_feature_engineer_does_not_change_row_count():
    df = pd.DataFrame(
        {
            "SK_ID_CURR": [10, 20],
            "AMT_INCOME_TOTAL": [50000.0, 70000.0],
            "AMT_CREDIT": [100000.0, 120000.0],
            "AMT_ANNUITY": [8000.0, 9000.0],
            "DAYS_EMPLOYED": [-100, -200],
            "DAYS_BIRTH": [-10000, -15000],
        }
    )
    out = FeatureEngineer().transform(df)
    assert len(out) == len(df)


def test_feature_engineer_ratios_are_finite_even_with_zero_or_nan_denominators():
    df = pd.DataFrame(
        {
            "SK_ID_CURR": [1, 2],
            "AMT_INCOME_TOTAL": [0.0, np.nan],
            "AMT_CREDIT": [100000.0, 200000.0],
            "AMT_ANNUITY": [5000.0, 10000.0],
            "AMT_GOODS_PRICE": [0.0, 150000.0],
            "CNT_FAM_MEMBERS": [0.0, np.nan],
            "DAYS_EMPLOYED": [-1000, FeatureEngineer.DAYS_EMPLOYED_ANOM],
            "DAYS_BIRTH": [-12000, -15000],
            "EXT_SOURCE_2": [0.3, 0.5],  # at least one ext source col so it doesn't crash
        }
    )
    out = FeatureEngineer().transform(df)

    ratio_cols = [
        "CREDIT_INCOME_RATIO",
        "ANNUITY_INCOME_RATIO",
        "EMPLOYED_TO_AGE",
        "APPS_ANNUITY_CREDIT_RATIO",
        "APPS_GOODS_CREDIT_RATIO",
        "APPS_ANNUITY_INCOME_RATIO",
        "APPS_CREDIT_INCOME_RATIO",
        "APPS_GOODS_INCOME_RATIO",
        "APPS_CNT_FAM_INCOME_RATIO",
    ]
    for col in ratio_cols:
        if col in out.columns:
            assert np.isfinite(out[col]).all(), f"Non-finite values in {col}"


def test_feature_engineer_age_and_employment_years_non_negative():
    df = pd.DataFrame(
        {
            "SK_ID_CURR": [1],
            "AMT_INCOME_TOTAL": [100000.0],
            "AMT_CREDIT": [200000.0],
            "AMT_ANNUITY": [20000.0],
            "DAYS_EMPLOYED": [-3650],
            "DAYS_BIRTH": [-36525],
        }
    )
    out = FeatureEngineer().transform(df)
    assert out.loc[0, "AGE_YEARS"] >= 0
    assert out.loc[0, "EMPLOYMENT_YEARS"] >= 0


def test_feature_engineer_does_not_mutate_input_df():
    df = pd.DataFrame(
        {
            "SK_ID_CURR": [1, 2],
            "AMT_INCOME_TOTAL": [100000.0, 50000.0],
            "AMT_CREDIT": [200000.0, 150000.0],
            "AMT_ANNUITY": [20000.0, 12000.0],
            "DAYS_EMPLOYED": [-1000, -2000],
            "DAYS_BIRTH": [-12000, -15000],
        }
    )

    df_before = df.copy(deep=True)
    _ = FeatureEngineer().transform(df)
    pd.testing.assert_frame_equal(df, df_before)


def test_feature_engineer_does_not_crash_when_optional_columns_missing():
    # only the minimal core columns
    df = pd.DataFrame(
        {
            "SK_ID_CURR": [1],
            "AMT_INCOME_TOTAL": [100000.0],
            "AMT_CREDIT": [200000.0],
            "AMT_ANNUITY": [20000.0],
            "DAYS_EMPLOYED": [-1000],
            "DAYS_BIRTH": [-12000],
        }
    )
    out = FeatureEngineer().transform(df)
    assert isinstance(out, pd.DataFrame)
    assert len(out) == 1
