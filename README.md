# Home Credit Default Risk

Binary classification — predicting whether a loan applicant will default, using the
[Home Credit Default Risk](https://www.kaggle.com/competitions/home-credit-default-risk) dataset.

## Results

All models evaluated on the same held-out test set (15% stratified split, never seen during training or SMOTE).

| Model          | ROC-AUC | Gini   | KS     | Brier  |
| -------------- | ------- | ------ | ------ | ------ |
| XGBoost        | 0.7809  | 0.5618 | 0.4228 | 0.0664 |
| FT-Transformer | 0.7415  | 0.4830 | 0.3744 | 0.0827 |
| TabNet         | 0.7344  | 0.4687 | 0.3573 | 0.0771 |

Metrics explained:

- **ROC-AUC** — probability that model ranks a defaulter above a non-defaulter
- **Gini** — `2 × AUC − 1`, standard in credit risk
- **KS** — maximum separation between default and non-default score distributions
- **Brier** — mean squared error of predicted probabilities (lower is better)

Training used SMOTE to balance the 8% default rate in the training split only.
All plots (ROC, Precision-Recall, loss curves) are in `plots/`.

## Setup

Requires Python 3.12+ and [uv](https://docs.astral.sh/uv/).

```bash
git clone https://github.com/Hamza-Emin/home-credit-default-risk
cd home-credit-default-risk
uv sync
dvc pull
```

`dvc pull` downloads the pre-merged feature matrix (`merged_train.csv`), raw CSVs, and
the fixed split indices — no preprocessing step needed.

DVC remotes (Google Drive, public viewer access):

- `data_store` — merged + raw data (~2.6 GB)
- `models_store` — trained model artifacts

## Train

Start the MLflow tracking server in a separate terminal (keep it running):

```bash
uv run mlflow server --host 127.0.0.1 --port 8080
```

Then train each model:

```bash
uv run python main.py train-xgboost --pull=False
uv run python main.py train-tabnet --pull=False
uv run python main.py train-ft-transformer --pull=False
```

View experiment results at `http://127.0.0.1:8080`.

## Reproducibility

The 70/15/15 stratified split is computed once and saved to
`data_folder/split_indices.npz` (SHA256 hash in `data_folder/split_indices_hash.txt`).
Every subsequent run loads the saved indices — no new split is created.

SMOTE is applied to the training split only, using `random_state=42`.
Because the split indices are fixed and the random seed is fixed,
SMOTE produces the same synthetic samples on every run.
Val and test sets are never touched by SMOTE and reflect the original class distribution (~8% default).

## Project structure

```text
configs/          Hydra config files (data, model, training, logging)
home_credit_risk/ Python package
  data/           data loading, splitting, SMOTE
  training/       XGBoost, TabNet, FT-Transformer trainers
plots/            ROC, PR, and loss curves for each model
models/           Saved model artifacts (DVC-tracked)
data_folder/      Raw CSVs + merged_train.csv (DVC-tracked)
```
