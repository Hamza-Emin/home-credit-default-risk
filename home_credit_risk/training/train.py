from omegaconf import DictConfig


def run_training(cfg: DictConfig) -> None:
    from home_credit_risk.data.loader import load_data

    print(f"Loading data from: {cfg.data.data_path}")
    data = load_data(cfg)
    print(
        f"Splits — train: {len(data['X_train'])} | "
        f"val: {len(data['X_val'])} | "
        f"test: {len(data['X_test'])}"
    )

    model_name = cfg.model.name
    print(f"Training model: {model_name}")

    if model_name == "xgboost":
        from home_credit_risk.training.xgboost_trainer import train_xgboost

        train_xgboost(data, cfg)

    elif model_name == "tabnet":
        from home_credit_risk.training.tabnet_trainer import train_tabnet

        train_tabnet(data, cfg)

    elif model_name == "ft_transformer":
        from home_credit_risk.training.ft_transformer_trainer import (
            train_ft_transformer,
        )

        train_ft_transformer(data, cfg)

    else:
        raise ValueError(
            f"Unknown model '{model_name}'. "
            "Choose from: xgboost, tabnet, ft_transformer"
        )
