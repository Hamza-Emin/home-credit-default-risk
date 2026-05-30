import subprocess
from pathlib import Path

import matplotlib.pyplot as plt
import mlflow
import numpy as np
import torch
from omegaconf import DictConfig
from pytorch_tabnet.tab_model import TabNetClassifier
from scipy.stats import ks_2samp
from sklearn.metrics import (
    brier_score_loss,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)
from sklearn.preprocessing import StandardScaler


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


def _save_plots(y_true: np.ndarray, proba: np.ndarray, loss_history: list) -> None:
    plots_dir = Path("plots")
    plots_dir.mkdir(exist_ok=True)

    # Training loss curve
    if loss_history:
        fig, ax = plt.subplots()
        ax.plot(loss_history)
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss")
        ax.set_title("TabNet — training loss")
        fig.tight_layout()
        fig.savefig(plots_dir / "tabnet_loss_curve.png", dpi=100)
        plt.close(fig)
        mlflow.log_artifact(str(plots_dir / "tabnet_loss_curve.png"))

    # ROC curve
    fpr, tpr, _ = roc_curve(y_true, proba)
    auc_val = roc_auc_score(y_true, proba)
    fig, ax = plt.subplots()
    ax.plot(fpr, tpr, label=f"AUC = {auc_val:.4f}")
    ax.plot([0, 1], [0, 1], "k--")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curve — TabNet")
    ax.legend()
    fig.tight_layout()
    fig.savefig(plots_dir / "tabnet_roc_curve.png", dpi=100)
    plt.close(fig)
    mlflow.log_artifact(str(plots_dir / "tabnet_roc_curve.png"))

    # Precision-Recall curve
    precision, recall, _ = precision_recall_curve(y_true, proba)
    fig, ax = plt.subplots()
    ax.plot(recall, precision)
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Precision-Recall Curve — TabNet")
    fig.tight_layout()
    fig.savefig(plots_dir / "tabnet_pr_curve.png", dpi=100)
    plt.close(fig)
    mlflow.log_artifact(str(plots_dir / "tabnet_pr_curve.png"))


def train_tabnet(data: dict, cfg: DictConfig) -> None:
    X_train_df = data["X_train"]
    y_train = data["y_train"]
    X_val_df = data["X_val"]
    y_val = data["y_val"]
    X_test_df = data["X_test"]
    y_test = data["y_test"]
    cat_cols = data["cat_cols"]
    cont_cols = data["cont_cols"]

    # Scale continuous, keep categoricals as int for embedding
    scaler = StandardScaler()
    X_train_cont = scaler.fit_transform(X_train_df[cont_cols].values.astype(np.float32))
    X_val_cont = scaler.transform(X_val_df[cont_cols].values.astype(np.float32))
    X_test_cont = scaler.transform(X_test_df[cont_cols].values.astype(np.float32))

    X_train_cat = X_train_df[cat_cols].values.astype(np.int64)
    X_val_cat = X_val_df[cat_cols].values.astype(np.int64)
    X_test_cat = X_test_df[cat_cols].values.astype(np.int64)

    X_train_np = np.hstack([X_train_cont, X_train_cat])
    X_val_np = np.hstack([X_val_cont, X_val_cat])
    X_test_np = np.hstack([X_test_cont, X_test_cat])

    # Categorical metadata for TabNet embeddings
    cat_idxs = list(range(len(cont_cols), len(cont_cols) + len(cat_cols)))
    cat_dims = [int(X_train_df[col].nunique()) + 1 for col in cat_cols]

    device = (
        "cuda"
        if (torch.cuda.is_available() and cfg.training.accelerator != "cpu")
        else "cpu"
    )

    mlflow.set_tracking_uri(cfg.logging.mlflow_uri)
    mlflow.set_experiment(cfg.logging.experiment_name)

    with mlflow.start_run(run_name="tabnet"):
        mlflow.log_params(
            {
                "model": "tabnet",
                "n_d": cfg.model.n_d,
                "n_a": cfg.model.n_a,
                "n_steps": cfg.model.n_steps,
                "gamma": cfg.model.gamma,
                "cat_emb_dim": cfg.model.cat_emb_dim,
                "learning_rate": cfg.model.learning_rate,
                "max_epochs": cfg.model.max_epochs,
                "batch_size": cfg.model.batch_size,
                "patience": cfg.model.patience,
                "git_commit": _git_commit(),
            }
        )

        model = TabNetClassifier(
            n_d=cfg.model.n_d,
            n_a=cfg.model.n_a,
            n_steps=cfg.model.n_steps,
            gamma=cfg.model.gamma,
            cat_idxs=cat_idxs,
            cat_dims=cat_dims,
            cat_emb_dim=cfg.model.cat_emb_dim,
            optimizer_params={"lr": cfg.model.learning_rate},
            scheduler_fn=torch.optim.lr_scheduler.CosineAnnealingLR,
            scheduler_params={"T_max": cfg.model.max_epochs},
            mask_type="entmax",
            device_name=device,
            verbose=20,
        )

        model.fit(
            X_train_np,
            y_train,
            eval_set=[(X_val_np, y_val)],
            eval_metric=["auc"],
            max_epochs=cfg.model.max_epochs,
            patience=cfg.model.patience,
            batch_size=cfg.model.batch_size,
            virtual_batch_size=256,
            weights=1,
        )

        # Log per-epoch metrics to MLflow
        try:
            loss_history = list(model.history["loss"])
        except KeyError:
            loss_history = []
        for step, loss_val in enumerate(loss_history):
            mlflow.log_metric("train_loss", float(loss_val), step=step)
        try:
            val_auc_history = list(model.history["val_0_auc"])
        except KeyError:
            val_auc_history = []
        for step, auc_val in enumerate(val_auc_history):
            mlflow.log_metric("val_auc", float(auc_val), step=step)

        proba = model.predict_proba(X_test_np)[:, 1]
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

        _save_plots(y_test, proba, loss_history)

        model_path = Path("models") / "tabnet"
        model_path.mkdir(parents=True, exist_ok=True)
        model.save_model(str(model_path / "model"))
        mlflow.log_artifact(str(model_path), artifact_path="model")

        print(
            f"\nTabNet | AUC: {auc:.4f} | Gini: {gini:.4f} "
            f"| KS: {ks:.4f} | Brier: {brier:.4f} | F1: {f1:.4f}"
        )
