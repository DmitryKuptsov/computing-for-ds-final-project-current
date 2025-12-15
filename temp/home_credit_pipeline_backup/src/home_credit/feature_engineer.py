import numpy as np
import pandas as pd


class FeatureEngineer:
    """Domain-specific feature engineering for credit risk."""

    DAYS_EMPLOYED_ANOM = 365243

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()

        # Missing flags
        if "AMT_INCOME_TOTAL" in out.columns:
            out["INCOME_MISSING"] = out["AMT_INCOME_TOTAL"].isna().astype(int)
        if "AMT_ANNUITY" in out.columns:
            out["ANNUITY_MISSING"] = out["AMT_ANNUITY"].isna().astype(int)

        # Handle known anomaly
        if "DAYS_EMPLOYED" in out.columns:
            out["DAYS_EMPLOYED_ANOM"] = (out["DAYS_EMPLOYED"] == self.DAYS_EMPLOYED_ANOM).astype(int)
            days_emp = out["DAYS_EMPLOYED"].replace(self.DAYS_EMPLOYED_ANOM, np.nan)
        else:
            days_emp = None

        # Base columns (only if present)
        income = out["AMT_INCOME_TOTAL"] if "AMT_INCOME_TOTAL" in out.columns else None
        credit = out["AMT_CREDIT"] if "AMT_CREDIT" in out.columns else None
        annuity = out["AMT_ANNUITY"] if "AMT_ANNUITY" in out.columns else None
        goods = out["AMT_GOODS_PRICE"] if "AMT_GOODS_PRICE" in out.columns else None
        fam = out["CNT_FAM_MEMBERS"] if "CNT_FAM_MEMBERS" in out.columns else None
        days_birth = out["DAYS_BIRTH"] if "DAYS_BIRTH" in out.columns else None

        def safe_div(num: pd.Series, den: pd.Series) -> pd.Series:
            den = den.replace(0, np.nan)
            return (num / den).replace([np.inf, -np.inf], np.nan).fillna(0)

        # Ratios (existing)
        if income is not None and credit is not None:
            out["CREDIT_INCOME_RATIO"] = safe_div(credit, income)

        if income is not None and annuity is not None:
            out["ANNUITY_INCOME_RATIO"] = safe_div(annuity, income)

        # Age (DAYS_BIRTH is negative)
        if days_birth is not None:
            out["AGE_YEARS"] = (-days_birth) / 365.25
            out["AGE_YEARS"] = out["AGE_YEARS"].clip(lower=0).fillna(0)

        # Employment years
        if days_emp is not None:
            emp_years = (-days_emp.clip(upper=0)) / 365.25
            out["EMPLOYMENT_YEARS"] = emp_years.fillna(0)

        # Employed-to-age ratio
        if "EMPLOYMENT_YEARS" in out.columns and "AGE_YEARS" in out.columns:
            out["EMPLOYED_TO_AGE"] = safe_div(out["EMPLOYMENT_YEARS"], out["AGE_YEARS"])

        # ---- Added "apps_*" feature set (simple and common) ----

        # EXT_SOURCE aggregated features
        ext_cols = [c for c in ["EXT_SOURCE_1", "EXT_SOURCE_2", "EXT_SOURCE_3"] if c in out.columns]
        if len(ext_cols) >= 2:
            out["APPS_EXT_SOURCE_MEAN"] = out[ext_cols].mean(axis=1)
            out["APPS_EXT_SOURCE_STD"] = out[ext_cols].std(axis=1)
            # fill std NaNs with global mean to keep it numeric
            out["APPS_EXT_SOURCE_STD"] = out["APPS_EXT_SOURCE_STD"].fillna(out["APPS_EXT_SOURCE_STD"].mean())
        elif len(ext_cols) == 1:
            # if only one source exists, mean = that value, std = 0
            out["APPS_EXT_SOURCE_MEAN"] = out[ext_cols[0]]
            out["APPS_EXT_SOURCE_STD"] = 0.0

        # Credit-based ratios
        if annuity is not None and credit is not None:
            out["APPS_ANNUITY_CREDIT_RATIO"] = safe_div(annuity, credit)
        if goods is not None and credit is not None:
            out["APPS_GOODS_CREDIT_RATIO"] = safe_div(goods, credit)

        # Income-based ratios
        if annuity is not None and income is not None:
            out["APPS_ANNUITY_INCOME_RATIO"] = safe_div(annuity, income)
        if credit is not None and income is not None:
            out["APPS_CREDIT_INCOME_RATIO"] = safe_div(credit, income)
        if goods is not None and income is not None:
            out["APPS_GOODS_INCOME_RATIO"] = safe_div(goods, income)

        # Income per family member
        if income is not None and fam is not None:
            out["APPS_CNT_FAM_INCOME_RATIO"] = safe_div(income, fam)

        return out
