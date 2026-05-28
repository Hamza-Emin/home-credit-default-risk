from pathlib import Path

import pandas as pd


def aggregate_bureau(data_path: Path) -> pd.DataFrame:
    df_bureau = pd.read_csv(data_path / "bureau.csv", encoding="unicode_escape")

    bureau_agg = df_bureau.groupby("SK_ID_CURR")["SK_ID_BUREAU"].count().reset_index()
    bureau_agg = bureau_agg.rename(columns={"SK_ID_BUREAU": "BUREAU_COUNT"})

    credit_type_dummies = pd.get_dummies(df_bureau["CREDIT_TYPE"])
    credit_type_dummies["SK_ID_CURR"] = df_bureau["SK_ID_CURR"]
    credit_type_counts = credit_type_dummies.groupby("SK_ID_CURR").sum().reset_index()
    bureau_agg = bureau_agg.merge(credit_type_counts, on="SK_ID_CURR", how="left")

    credit_active_dummies = pd.get_dummies(df_bureau["CREDIT_ACTIVE"])
    credit_active_dummies["SK_ID_CURR"] = df_bureau["SK_ID_CURR"]
    credit_active_counts = (
        credit_active_dummies.groupby("SK_ID_CURR").sum().reset_index()
    )
    bureau_agg = bureau_agg.merge(credit_active_counts, on="SK_ID_CURR", how="left")

    numeric_aggs = {
        "AMT_CREDIT_SUM": "sum",
        "AMT_CREDIT_SUM_DEBT": "sum",
        "AMT_CREDIT_SUM_OVERDUE": "sum",
        "AMT_CREDIT_MAX_OVERDUE": "max",
        "CREDIT_DAY_OVERDUE": "sum",
        "CNT_CREDIT_PROLONG": "sum",
        "DAYS_CREDIT": "max",
    }

    for col, func in numeric_aggs.items():
        agg_df = df_bureau.groupby("SK_ID_CURR")[col].agg(func).reset_index()
        bureau_agg = bureau_agg.merge(agg_df, on="SK_ID_CURR", how="left")

    return bureau_agg
