from pathlib import Path

import pandas as pd

from home_credit_risk.data_aggregation.bureau import aggregate_bureau
from home_credit_risk.data_aggregation.bureau_balance import aggregate_bureau_balance
from home_credit_risk.data_aggregation.credit_card import aggregate_credit_card
from home_credit_risk.data_aggregation.installments import aggregate_installments
from home_credit_risk.data_aggregation.pos_cash import aggregate_pos_cash
from home_credit_risk.data_aggregation.previous import aggregate_previous


def merge_all(data_path: Path) -> pd.DataFrame:
    app_train = pd.read_csv(data_path / "application_train.csv")
    print(f"application_train loaded — shape: {app_train.shape}")

    agg_tables = [
        ("bureau", aggregate_bureau(data_path)),
        ("previous", aggregate_previous(data_path)),
        ("credit_card", aggregate_credit_card(data_path)),
        ("pos_cash", aggregate_pos_cash(data_path)),
        ("installments", aggregate_installments(data_path)),
        ("bureau_balance", aggregate_bureau_balance(data_path)),
    ]

    for name, agg_df in agg_tables:
        app_train = app_train.merge(agg_df, on="SK_ID_CURR", how="left")
        print(f"  merged {name:20s} → shape: {app_train.shape}")

    num_cols = app_train.select_dtypes(include="number").columns
    app_train[num_cols] = app_train[num_cols].fillna(0)

    str_cols = app_train.select_dtypes(include="object").columns
    for col in str_cols:
        app_train[col] = app_train[col].fillna(app_train[col].mode()[0])

    return app_train
