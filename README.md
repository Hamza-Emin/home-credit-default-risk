# Home Credit Default Risk

## Project Description

### Problem Statement

The goal of this project is to predict whether a loan applicant will experience repayment
difficulties, based on their application data, credit history, and behavioral financial
records. This is a binary classification problem: the target variable is 1 if the client
had late payments exceeding a threshold on at least one of the first installments of their
loan, and 0 otherwise.

This problem is commercially critical for consumer lending institutions. Incorrectly
rejecting a creditworthy applicant results in lost business, while approving a high-risk
applicant leads to direct financial losses. Home Credit Group released this dataset to
improve loan decisions for the unbanked population — people who lack traditional credit
histories. A reliable default risk predictor allows institutions to make fairer,
data-driven, and financially sound lending decisions.

### Dataset

The [Home Credit Default Risk](https://www.kaggle.com/competitions/home-credit-default-risk)
dataset consists of 8 CSV files (~2.6 GB total). The main file is `application_train.csv`
with 307,511 rows and 122 columns (106 numerical, 16 categorical) covering demographics,
employment, loan characteristics, external credit scores, and regional indicators.

The remaining 7 files contain historical data per applicant: previous loan applications
(1.67M rows), credit bureau records (1.71M rows), installment payment history (13.6M rows),
credit card balances (3.84M rows), and more. These are aggregated and joined onto the main
table as additional features.

Key challenge: severe class imbalance (~8% defaulters). Handled via SMOTE on the training
split only.

### Models

| Model          | ROC-AUC | Gini   | KS     | Brier  | F1 (opt) |
| -------------- | ------- | ------ | ------ | ------ | -------- |
| XGBoost        | 0.7809  | 0.5618 | 0.4228 | 0.0664 | 0.3367   |
| FT-Transformer | 0.7415  | 0.4830 | 0.3744 | 0.0827 | 0.2807   |
| TabNet         | 0.7344  | 0.4687 | 0.3573 | 0.0771 | 0.1950   |

- **XGBoost** — gradient boosting baseline with `scale_pos_weight` for class imbalance
- **TabNet** — attention-based tabular model with automatic feature selection
- **FT-Transformer** — transformer architecture applied to tabular features

### Metrics

- **ROC-AUC** — official competition metric; robust to class imbalance
- **Gini** — `2 × AUC − 1`; standard in credit risk / banking industry
- **KS Statistic** — maximum separation between predicted default and non-default distributions
- **Brier Score** — mean squared error of predicted probabilities; measures calibration
- **F1 (optimal threshold)** — F1 at the threshold that maximises it on the test set

### Validation

70 / 15 / 15 stratified split saved to `data_folder/split_indices.npz` (SHA256 hash in
`data_folder/split_indices_hash.txt`). SMOTE is applied to the training split only at
training time — val and test sets reflect the original class distribution.

---

## Setup

Requires Python 3.12+ and [uv](https://docs.astral.sh/uv/).

```bash
git clone https://github.com/Hamza-Emin/home-credit-default-risk
cd home-credit-default-risk
uv sync
uv run pre-commit install
uv run dvc pull
```

`dvc pull` downloads the pre-merged feature matrix (`merged_train.csv`), raw CSVs, and
the fixed split indices — no preprocessing step needed.

> All commands use the `uv run` prefix so they work without manually activating the virtual
> environment. Alternatively, activate it first with `.venv\Scripts\activate` (Windows) or
> `source .venv/bin/activate` (Linux/macOS) and omit the `uv run` prefix.

DVC remotes (Google Drive, public viewer access):

- `data_store` — merged + raw data (~2.6 GB)
- `models_store` — trained model artifacts

## Train

> **GPU note:** Training was done on an NVIDIA RTX 5080 (16 GB VRAM). All three models
> default to GPU. If you do not have a CUDA-capable GPU, add `--gpu=False` and training
> will fall back to CPU automatically (TabNet and FT-Transformer will be significantly
> slower).

Start the MLflow tracking server in a separate terminal (keep it running):

```bash
uv run mlflow server --host 127.0.0.1 --port 8080
```

Then train each model (in a second terminal):

```bash
# with GPU (default)
uv run python main.py train-xgboost --pull=False
uv run python main.py train-tabnet --pull=False
uv run python main.py train-ft-transformer --pull=False

# without GPU
uv run python main.py train-xgboost --pull=False --gpu=False
uv run python main.py train-tabnet --pull=False --gpu=False
uv run python main.py train-ft-transformer --pull=False --gpu=False
```

View experiment results at `http://127.0.0.1:8080`.

## Project Structure

```text
configs/          Hydra config files (data, model, training, logging)
home_credit_risk/ Python package
  data/           data loading, splitting, SMOTE
  data_aggregation/ feature engineering from supplementary tables
  training/       XGBoost, TabNet, FT-Transformer trainers
plots/            ROC, PR, and loss curves for each model
models/           Saved model artifacts (DVC-tracked)
data_folder/      Raw CSVs + merged_train.csv (DVC-tracked)
```
