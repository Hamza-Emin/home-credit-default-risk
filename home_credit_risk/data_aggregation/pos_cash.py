from pathlib import Path

import pandas as pd


def aggregate_pos_cash(data_path: Path) -> pd.DataFrame:
    df_pos = pd.read_csv(data_path / "POS_CASH_balance.csv", encoding="unicode_escape")

    pos_agg = df_pos.groupby("SK_ID_CURR")["SK_ID_PREV"].count().reset_index()
    pos_agg = pos_agg.rename(columns={"SK_ID_PREV": "POS_COUNT"})

    numeric_aggs = {
        "SK_DPD": ["max", "mean"],
        "SK_DPD_DEF": ["max", "mean"],
        "CNT_INSTALMENT": "mean",
        "CNT_INSTALMENT_FUTURE": "mean",
    }

    for col, func in numeric_aggs.items():
        agg_df = df_pos.groupby("SK_ID_CURR")[col].agg(func).reset_index()
        if isinstance(func, list):
            agg_df.columns = ["SK_ID_CURR"] + [f"{col}_{f.upper()}" for f in func]
        pos_agg = pos_agg.merge(agg_df, on="SK_ID_CURR", how="left")

    contract_status_dummies = pd.get_dummies(df_pos["NAME_CONTRACT_STATUS"])
    contract_status_dummies["SK_ID_CURR"] = df_pos["SK_ID_CURR"]
    contract_status_counts = (
        contract_status_dummies.groupby("SK_ID_CURR").sum().reset_index()
    )
    pos_agg = pos_agg.merge(contract_status_counts, on="SK_ID_CURR", how="left")

    return pos_agg
