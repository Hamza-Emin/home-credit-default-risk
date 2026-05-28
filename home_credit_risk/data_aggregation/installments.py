from pathlib import Path

import pandas as pd


def aggregate_installments(data_path: Path) -> pd.DataFrame:
    df_inst = pd.read_csv(
        data_path / "installments_payments.csv", encoding="unicode_escape"
    )

    inst_agg = df_inst.groupby("SK_ID_CURR")["SK_ID_PREV"].count().reset_index()
    inst_agg = inst_agg.rename(columns={"SK_ID_PREV": "INST_COUNT"})

    # positive = paid late, negative = paid early
    df_inst["DAYS_LATE"] = df_inst["DAYS_ENTRY_PAYMENT"] - df_inst["DAYS_INSTALMENT"]
    # < 1 means underpayment
    df_inst["PAYMENT_RATIO"] = df_inst["AMT_PAYMENT"] / df_inst["AMT_INSTALMENT"]

    numeric_aggs = {
        "DAYS_LATE": ["mean", "max"],
        "PAYMENT_RATIO": "mean",
        "AMT_INSTALMENT": "mean",
        "AMT_PAYMENT": "mean",
    }

    for col, func in numeric_aggs.items():
        agg_df = df_inst.groupby("SK_ID_CURR")[col].agg(func).reset_index()
        if isinstance(func, list):
            agg_df.columns = ["SK_ID_CURR"] + [f"{col}_{f.upper()}" for f in func]
        inst_agg = inst_agg.merge(agg_df, on="SK_ID_CURR", how="left")

    return inst_agg
