from pathlib import Path

import pandas as pd


def aggregate_credit_card(data_path: Path) -> pd.DataFrame:
    df_cc = pd.read_csv(
        data_path / "credit_card_balance.csv", encoding="unicode-escape"
    )

    cc_agg = df_cc.groupby("SK_ID_CURR")["SK_ID_PREV"].count().reset_index()
    cc_agg = cc_agg.rename(columns={"SK_ID_PREV": "CC_COUNT"})

    df_cc["CREDIT_UTILIZATION"] = (
        df_cc["AMT_BALANCE"] / df_cc["AMT_CREDIT_LIMIT_ACTUAL"]
    )

    numeric_aggs = {
        "SK_DPD": ["max", "mean"],
        "SK_DPD_DEF": ["max", "mean"],
        "AMT_BALANCE": "mean",
        "AMT_CREDIT_LIMIT_ACTUAL": "mean",
        "AMT_DRAWINGS_ATM_CURRENT": "mean",
        "AMT_DRAWINGS_CURRENT": "mean",
        "AMT_PAYMENT_CURRENT": "mean",
        "AMT_PAYMENT_TOTAL_CURRENT": "mean",
        "AMT_RECEIVABLE_PRINCIPAL": "mean",
        "CREDIT_UTILIZATION": "mean",
    }

    for col, func in numeric_aggs.items():
        agg_df = df_cc.groupby("SK_ID_CURR")[col].agg(func).reset_index()
        if isinstance(func, list):
            agg_df.columns = ["SK_ID_CURR"] + [f"{col}_{f.upper()}" for f in func]
        cc_agg = cc_agg.merge(agg_df, on="SK_ID_CURR", how="left")

    contract_status_dummies = pd.get_dummies(df_cc["NAME_CONTRACT_STATUS"])
    contract_status_dummies["SK_ID_CURR"] = df_cc["SK_ID_CURR"]
    contract_status_counts = (
        contract_status_dummies.groupby("SK_ID_CURR").sum().reset_index()
    )
    cc_agg = cc_agg.merge(contract_status_counts, on="SK_ID_CURR", how="left")

    return cc_agg
