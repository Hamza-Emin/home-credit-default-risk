import subprocess
from pathlib import Path

import matplotlib.pyplot as plt
import mlflow
import numpy as np
import torch
from omegaconf import DictConfig
from pytorch_lightning.callbacks import Callback
from pytorch_tabular import TabularModel
from pytorch_tabular.config import DataConfig, OptimizerConfig, TrainerConfig
from pytorch_tabular.models import FTTransformerConfig
from scipy.stats import ks_2samp
from sklearn.metrics import (
    brier_score_loss,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)


def _git_commit() -> str:
    try:
        return (
            subprocess.check_output(
                ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL
            )
            .decode()
            .strip()
        )
    except Exception:
        return "unknown"


class _MLflowEpochLogger(Callback):
    def __init__(self):
        self.train_losses: list = []
        self.val_losses: list = []

    def on_train_epoch_end(self, trainer, pl_module):
        metrics = trainer.callback_metrics
        step = trainer.current_epoch
        for key, value in metrics.items():
            try:
                mlflow.log_metric(key, float(value), step=step)
            except Exception:
                pass
        if "train_loss" in metrics:
            self.train_losses.append(float(metrics["train_loss"]))
        if "valid_loss" in metrics:
            self.val_losses.append(float(metrics["valid_loss"]))


def _save_plots(
    y_true: np.ndarray,
    proba: np.ndarray,
    train_losses: list,
    val_losses: list,
) -> None:
    plots_dir = Path("plots")
    plots_dir.mkdir(exist_ok=True)

    # Train vs validation loss curve
    if train_losses:
        fig, ax = plt.subplots()
        ax.plot(train_losses, label="train")
        if val_losses:
            ax.plot(val_losses, label="val")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss")
        ax.set_title("FT-Transformer — loss per epoch")
        ax.legend()
        fig.tight_layout()
        fig.savefig(plots_dir / "ft_transformer_loss_curve.png", dpi=100)
        plt.close(fig)
        mlflow.log_artifact(str(plots_dir / "ft_transformer_loss_curve.png"))

    # ROC curve
    fpr, tpr, _ = roc_curve(y_true, proba)
    auc_val = roc_auc_score(y_true, proba)
    fig, ax = plt.subplots()
    ax.plot(fpr, tpr, label=f"AUC = {auc_val:.4f}")
    ax.plot([0, 1], [0, 1], "k--")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curve — FT-Transformer")
    ax.legend()
    fig.tight_layout()
    fig.savefig(plots_dir / "ft_transformer_roc_curve.png", dpi=100)
    plt.close(fig)
    mlflow.log_artifact(str(plots_dir / "ft_transformer_roc_curve.png"))

    # Precision-Recall curve
    precision, recall, _ = precision_recall_curve(y_true, proba)
    fig, ax = plt.subplots()
    ax.plot(recall, precision)
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Precision-Recall Curve — FT-Transformer")
    fig.tight_layout()
    fig.savefig(plots_dir / "ft_transformer_pr_curve.png", dpi=100)
    plt.close(fig)
    mlflow.log_artifact(str(plots_dir / "ft_transformer_pr_curve.png"))


def train_ft_transformer(data: dict, cfg: DictConfig) -> None:
    X_train_df = data["X_train"].copy()
    X_val_df = data["X_val"].copy()
    X_test_df = data["X_test"].copy()
    y_train = data["y_train"]
    y_val = data["y_val"]
    y_test = data["y_test"]
    cat_cols = data["cat_cols"]
    cont_cols = data["cont_cols"]

    target_col = "TARGET"

    # pytorch_tabular requires categoricals as strings and continuous as float32
    for df in (X_train_df, X_val_df, X_test_df):
        for col in cat_cols:
            df[col] = df[col].astype(str)
        for col in cont_cols:
            df[col] = df[col].astype("float32")

    train_df = X_train_df.copy()
    val_df = X_val_df.copy()
    test_df = X_test_df.copy()
    train_df[target_col] = y_train.astype(int)
    val_df[target_col] = y_val.astype(int)
    test_df[target_col] = y_test.astype(int)

    torch.set_float32_matmul_precision("high")

    mlflow.set_tracking_uri(cfg.logging.mlflow_uri)
    mlflow.set_experiment(cfg.logging.experiment_name)

    with mlflow.start_run(run_name="ft-transformer"):
        mlflow.log_params(
            {
                "model": "ft_transformer",
                "num_attn_blocks": cfg.model.num_attn_blocks,
                "num_heads": cfg.model.num_heads,
                "attn_dropout": cfg.model.attn_dropout,
                "ff_dropout": cfg.model.ff_dropout,
                "learning_rate": cfg.model.learning_rate,
                "cat_emb_dim": cfg.model.cat_emb_dim,
                "max_epochs": cfg.training.max_epochs,
                "batch_size": cfg.training.batch_size,
                "patience": cfg.training.patience,
                "git_commit": _git_commit(),
            }
        )

        data_config = DataConfig(
            target=[target_col],
            continuous_cols=cont_cols,
            categorical_cols=cat_cols,
        )

        accelerator = cfg.training.accelerator if torch.cuda.is_available() else "cpu"
        use_gpu = accelerator == "gpu" and torch.cuda.is_available()
        precision = cfg.training.precision if use_gpu else 32

        trainer_config = TrainerConfig(
            max_epochs=cfg.training.max_epochs,
            batch_size=cfg.training.batch_size,
            accelerator=accelerator,
            devices=1,
            precision=precision,
            early_stopping="valid_accuracy",
            early_stopping_patience=cfg.training.patience,
            early_stopping_mode="max",
            progress_bar="simple",
        )

        optimizer_config = OptimizerConfig(
            lr_scheduler=cfg.training.lr_scheduler,
            lr_scheduler_params={"T_max": cfg.training.lr_scheduler_t_max},
        )

        model_config = FTTransformerConfig(
            task="classification",
            num_attn_blocks=cfg.model.num_attn_blocks,
            num_heads=cfg.model.num_heads,
            attn_dropout=cfg.model.attn_dropout,
            ff_dropout=cfg.model.ff_dropout,
            learning_rate=cfg.model.learning_rate,
        )

        epoch_logger = _MLflowEpochLogger()

        # pytorch_tabular checkpoints embed omegaconf objects; patch loader for trusted local files
        import pytorch_tabular.tabular_model as _pt_model

        def _patched_pl_load(path_or_url, map_location=None):
            return torch.load(
                path_or_url, map_location=map_location, weights_only=False
            )

        _pt_model.pl_load = _patched_pl_load

        ft_model = TabularModel(
            data_config=data_config,
            model_config=model_config,
            optimizer_config=optimizer_config,
            trainer_config=trainer_config,
        )

        ft_model.fit(
            train=train_df,
            validation=val_df,
            callbacks=[epoch_logger],
        )

        result = ft_model.predict(test_df)
        prob_col = [c for c in result.columns if "1_probability" in c]
        if not prob_col:
            raise KeyError(
                f"No *1_probability column found. Got: {list(result.columns)}"
            )
        proba = result[prob_col[0]].values

        auc = roc_auc_score(y_test, proba)
        gini = 2.0 * auc - 1.0
        ks = ks_2samp(proba[y_test == 1], proba[y_test == 0]).statistic
        brier = brier_score_loss(y_test, proba)
        precision_vals, recall_vals, thresholds = precision_recall_curve(y_test, proba)
        f1_vals = (
            2
            * precision_vals[:-1]
            * recall_vals[:-1]
            / (precision_vals[:-1] + recall_vals[:-1] + 1e-9)
        )
        f1 = float(f1_vals.max())

        mlflow.log_metrics(
            {
                "test_auc": auc,
                "test_gini": gini,
                "test_ks": ks,
                "test_brier": brier,
                "test_f1": f1,
            }
        )

        _save_plots(y_test, proba, epoch_logger.train_losses, epoch_logger.val_losses)

        model_path = Path("models") / "ft_transformer"
        model_path.mkdir(parents=True, exist_ok=True)
        ft_model.save_model(str(model_path))
        mlflow.log_artifact(str(model_path), artifact_path="model")

        print(
            f"\nFT-Transformer | AUC: {auc:.4f} | Gini: {gini:.4f} "
            f"| KS: {ks:.4f} | Brier: {brier:.4f} | F1: {f1:.4f}"
        )
