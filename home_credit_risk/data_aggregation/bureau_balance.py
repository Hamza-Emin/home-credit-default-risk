from pathlib import Path

import pandas as pd


def aggregate_bureau_balance(data_path: Path) -> pd.DataFrame:
    df_bureau = pd.read_csv(data_path / "bureau.csv", encoding="unicode_escape")
    df_bureau_balance = pd.read_csv(
        data_path / "bureau_balance.csv", encoding="unicode_escape"
    )

    # Two-step join: bureau_balance has SK_ID_BUREAU, bureau has SK_ID_BUREAU + SK_ID_CURR
    bb = df_bureau_balance.merge(
        df_bureau[["SK_ID_BUREAU", "SK_ID_CURR"]], on="SK_ID_BUREAU", how="left"
    )

    bb_agg = bb.groupby("SK_ID_CURR")["MONTHS_BALANCE"].count().reset_index()
    bb_agg = bb_agg.rename(columns={"MONTHS_BALANCE": "BB_MONTHS_COUNT"})

    # STATUS values: 0-5 = DPD buckets, C = closed, X = unknown
    status_dummies = pd.get_dummies(bb["STATUS"])
    status_dummies["SK_ID_CURR"] = bb["SK_ID_CURR"]
    status_counts = status_dummies.groupby("SK_ID_CURR").sum().reset_index()
    bb_agg = bb_agg.merge(status_counts, on="SK_ID_CURR", how="left")

    return bb_agg
