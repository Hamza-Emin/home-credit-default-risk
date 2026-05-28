from pathlib import Path

import pandas as pd


def aggregate_previous(data_path: Path) -> pd.DataFrame:
    df_previous = pd.read_csv(data_path / "previous_application.csv")

    prev_agg = df_previous.groupby("SK_ID_CURR")["SK_ID_PREV"].count().reset_index()
    prev_agg = prev_agg.rename(columns={"SK_ID_PREV": "PREV_APP_COUNT"})

    contract_status_dummies = pd.get_dummies(df_previous["NAME_CONTRACT_STATUS"])
    contract_status_dummies["SK_ID_CURR"] = df_previous["SK_ID_CURR"]
    contract_status_counts = (
        contract_status_dummies.groupby("SK_ID_CURR").sum().reset_index()
    )
    prev_agg = prev_agg.merge(contract_status_counts, on="SK_ID_CURR", how="left")

    numeric_aggs = {
        "AMT_APPLICATION": "mean",
        "AMT_CREDIT": "mean",
        "AMT_ANNUITY": "mean",
        "AMT_DOWN_PAYMENT": "mean",
        "CNT_PAYMENT": "mean",
        "DAYS_DECISION": "max",
    }

    for col, func in numeric_aggs.items():
        agg_df = df_previous.groupby("SK_ID_CURR")[col].agg(func).reset_index()
        prev_agg = prev_agg.merge(agg_df, on="SK_ID_CURR", how="left")

    return prev_agg
