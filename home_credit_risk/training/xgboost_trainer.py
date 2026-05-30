import subprocess
from pathlib import Path

import matplotlib.pyplot as plt
import mlflow
import mlflow.xgboost
import numpy as np
import xgboost as xgb
from omegaconf import DictConfig
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


class _XGBMLflowCallback(xgb.callback.TrainingCallback):
    def after_iteration(self, model, epoch, evals_log):
        for data_name, metric_dict in evals_log.items():
            for metric_name, values in metric_dict.items():
                mlflow.log_metric(
                    f"{data_name}_{metric_name}", float(values[-1]), step=epoch
                )
        return False


def _save_plots(y_true: np.ndarray, proba: np.ndarray, evals_result: dict) -> None:
    plots_dir = Path("plots")
    plots_dir.mkdir(exist_ok=True)

    # AUC per boosting round (train vs val)
    train_auc = evals_result.get("validation_0", {}).get("auc", [])
    val_auc = evals_result.get("validation_1", {}).get("auc", [])
    if val_auc:
        fig, ax = plt.subplots()
        ax.plot(train_auc, label="train")
        ax.plot(val_auc, label="val")
        ax.set_xlabel("Boosting round")
        ax.set_ylabel("AUC")
        ax.set_title("XGBoost — AUC per boosting round")
        ax.legend()
        fig.tight_layout()
        fig.savefig(plots_dir / "xgboost_auc_curve.png", dpi=100)
        plt.close(fig)
        mlflow.log_artifact(str(plots_dir / "xgboost_auc_curve.png"))

    # ROC curve on test set
    fpr, tpr, _ = roc_curve(y_true, proba)
    auc_val = roc_auc_score(y_true, proba)
    fig, ax = plt.subplots()
    ax.plot(fpr, tpr, label=f"AUC = {auc_val:.4f}")
    ax.plot([0, 1], [0, 1], "k--")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curve — XGBoost")
    ax.legend()
    fig.tight_layout()
    fig.savefig(plots_dir / "xgboost_roc_curve.png", dpi=100)
    plt.close(fig)
    mlflow.log_artifact(str(plots_dir / "xgboost_roc_curve.png"))

    # Precision-Recall curve
    precision, recall, _ = precision_recall_curve(y_true, proba)
    fig, ax = plt.subplots()
    ax.plot(recall, precision)
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Precision-Recall Curve — XGBoost")
    fig.tight_layout()
    fig.savefig(plots_dir / "xgboost_pr_curve.png", dpi=100)
    plt.close(fig)
    mlflow.log_artifact(str(plots_dir / "xgboost_pr_curve.png"))


def train_xgboost(data: dict, cfg: DictConfig) -> None:
    X_train = data["X_train"]
    y_train = data["y_train"]
    X_val = data["X_val"]
    y_val = data["y_val"]
    X_test = data["X_test"]
    y_test = data["y_test"]

    scale_pos_weight = float((y_train == 0).sum()) / float((y_train == 1).sum())

    mlflow.set_tracking_uri(cfg.logging.mlflow_uri)
    mlflow.set_experiment(cfg.logging.experiment_name)

    with mlflow.start_run(run_name="xgboost-baseline"):
        mlflow.log_params(
            {
                "model": "xgboost",
                "n_estimators": cfg.model.n_estimators,
                "max_depth": cfg.model.max_depth,
                "learning_rate": cfg.model.learning_rate,
                "early_stopping_rounds": cfg.model.early_stopping_rounds,
                "scale_pos_weight": round(scale_pos_weight, 4),
                "random_state": cfg.data.random_state,
                "git_commit": _git_commit(),
            }
        )

        use_gpu = cfg.training.accelerator != "cpu"
        model = xgb.XGBClassifier(
            n_estimators=cfg.model.n_estimators,
            max_depth=cfg.model.max_depth,
            learning_rate=cfg.model.learning_rate,
            objective="binary:logistic",
            eval_metric="auc",
            scale_pos_weight=scale_pos_weight,
            early_stopping_rounds=cfg.model.early_stopping_rounds,
            random_state=cfg.data.random_state,
            device="cuda" if use_gpu else "cpu",
            callbacks=[_XGBMLflowCallback()],
        )

        model.fit(
            X_train,
            y_train,
            eval_set=[(X_train, y_train), (X_val, y_val)],
            verbose=100,
        )

        proba = model.predict_proba(X_test)[:, 1]
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

        _save_plots(y_test, proba, model.evals_result())

        model_path = Path("models") / "xgboost"
        model_path.mkdir(parents=True, exist_ok=True)
        model.save_model(model_path / "model.ubj")
        mlflow.log_artifact(str(model_path / "model.ubj"), artifact_path="model")

        print(
            f"\nXGBoost | AUC: {auc:.4f} | Gini: {gini:.4f} "
            f"| KS: {ks:.4f} | Brier: {brier:.4f} | F1: {f1:.4f}"
        )
