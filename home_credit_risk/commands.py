import subprocess
from pathlib import Path

import fire
from hydra import compose, initialize_config_dir
from omegaconf import DictConfig


def _load_config(
    config_path: str = "configs", overrides: list[str] | None = None
) -> DictConfig:
    abs_config_path = str(Path(config_path).absolute())
    with initialize_config_dir(config_dir=abs_config_path, version_base=None):
        cfg = compose(config_name="config", overrides=overrides or [])
    return cfg


def _dvc_pull(target: str) -> None:
    print(f"Pulling {target} from DVC remote...")
    subprocess.run(["dvc", "pull", target], check=True)


def preprocess(
    config_path: str = "configs", pull: bool = True, override: str = ""
) -> None:
    overrides = [o.strip() for o in override.split(",") if o.strip()]
    cfg = _load_config(config_path, overrides)

    if pull:
        _dvc_pull("data_folder.dvc")

    data_path = Path(cfg.data.data_path)

    from home_credit_risk.data_aggregation.merging import merge_all

    merged = merge_all(data_path)
    output_path = data_path / "merged_train.csv"
    merged.to_csv(output_path, index=False)
    print(f"\nSaved → {output_path}")
    print(f"Final shape: {merged.shape}")


def train(config_path: str = "configs", pull: bool = True, override: str = "") -> None:
    overrides = [o.strip() for o in override.split(",") if o.strip()]
    cfg = _load_config(config_path, overrides)

    if pull:
        _dvc_pull("data_folder.dvc")

    from home_credit_risk.training.train import run_training

    run_training(cfg)


def train_xgboost(config_path: str = "configs", pull: bool = True) -> None:
    train(config_path=config_path, pull=pull, override="model=xgboost")


def train_tabnet(config_path: str = "configs", pull: bool = True) -> None:
    train(config_path=config_path, pull=pull, override="model=tabnet")


def train_ft_transformer(config_path: str = "configs", pull: bool = True) -> None:
    train(config_path=config_path, pull=pull, override="model=ft_transformer")


def main() -> None:
    fire.Fire(
        {
            "preprocess": preprocess,
            "train": train,
            "train-xgboost": train_xgboost,
            "train-tabnet": train_tabnet,
            "train-ft-transformer": train_ft_transformer,
        }
    )


if __name__ == "__main__":
    main()
