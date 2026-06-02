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

| Model          | ROC-AUC | Gini   | KS     | Brier  | F1     |
| -------------- | ------- | ------ | ------ | ------ | ------ |
| XGBoost        | 0.7794  | 0.5588 | 0.4210 | 0.0664 | 0.3356 |
| FT-Transformer | 0.7415  | 0.4830 | 0.3744 | 0.0827 | 0.2807 |
| TabNet         | 0.7344  | 0.4687 | 0.3573 | 0.0771 | 0.1950 |

- **XGBoost** — gradient boosting baseline with `scale_pos_weight` for class imbalance
- **TabNet** — attention-based tabular model with automatic feature selection
- **FT-Transformer** — transformer architecture applied to tabular features

### Metrics

- **ROC-AUC** — official competition metric; robust to class imbalance
- **Gini** — `2 × AUC − 1`; standard in credit risk / banking industry
- **KS Statistic** — maximum separation between predicted default and non-default distributions
- **Brier Score** — mean squared error of predicted probabilities; measures calibration
- **F1** — F1 at the threshold that maximises it on the test set

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

> Important Note: I used google drive as my sotrage , dvc pull directly pulls the data from my google drive. In order to achieve this, I put my google drive folder secrets into the project directly so dvc pull commands run without any additional effort.
> I know we are trying to do a project in the industrial level. Normally I know that , ı should share the secrets about google folder privatly to reciever and they would put it into correct places. But ı choose this becuase ı did not want you waste effort about creating credentials.
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

### Stage 1 — Data loading

`dvc pull` (run during Setup) downloads `data_folder/` which contains the raw CSVs,
the pre-merged feature matrix `merged_train.csv`, and the fixed split indices
`split_indices.npz`.

```bash
uv run dvc pull
```

### Stage 2 — Preprocessing

Aggregates the 7 supplementary tables (bureau, previous applications, installments,
credit card balances, POS cash) and merges them into a single flat feature matrix joined
on `SK_ID_CURR`. The result is saved as `data_folder/merged_train.csv`.

```bash
uv run python main.py preprocess --pull=False
```

> This step is already done — `merged_train.csv` is included in the DVC pull. Only run
> this if you modify the feature engineering code and need to regenerate the merged file.

### Stage 3 — MLflow server

Start the tracking server in a separate terminal and keep it running:

```bash
uv run mlflow server --host 127.0.0.1 --port 8080
```

### Stage 4 — Model training

Each model is trained independently. Run one or all in a second terminal:

**XGBoost** (baseline, ~5 min on GPU):

```bash
uv run python main.py train-xgboost --pull=False
```

**TabNet** (attention-based, ~15 min on GPU):

```bash
uv run python main.py train-tabnet --pull=False
```

**FT-Transformer** (transformer, ~30 min on GPU):

```bash
uv run python main.py train-ft-transformer --pull=False
```

To train without a GPU, append `--gpu=False` to any command above.

View all experiment results at `http://127.0.0.1:8080`. Plots are saved to `plots/`.

Note: I did not put an inference script from the models that we trained as I said in my project proposal becuase from the project document that you shared with us, it is being said that we are responsible for training the models , not the inference part.
If ı misunderstood , please let me know , I can put that quickly.

## Docker , 2nd Way To Run The Project After Pulling The Data with DVC PULL

I implemented the training and mlflow process also in the docker. The MLflow container and trainer container share a
network — the trainer automatically points to the MLflow container using the `MLFLOW_URI`
environment variable. If a CUDA-capable GPU and `nvidia-container-toolkit` are present the
trainer uses it; otherwise training falls back to CPU silently.

**Prerequisites:** [Docker](https://docs.docker.com/get-docker/) installed. For GPU support
also install [nvidia-container-toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html).

Build the image (first time only, or after dependency changes):

```bash
docker compose build
```

Start MLflow and train a model and save the model , plots into appropiete folders :

```bash
# MLflow server starts automatically as a dependency
docker compose run --rm trainer uv run python main.py train-xgboost --pull=False
docker compose run --rm trainer uv run python main.py train-tabnet --pull=False
docker compose run --rm trainer uv run python main.py train-ft-transformer --pull=False
```

View experiment results at `http://localhost:8080` (MLflow stays running in the background).

Stop all containers:

```bash
docker compose down
```

> Data (`data_folder/`), models (`models/`), plots (`plots/`), and MLflow runs (`mlruns/`)
> are mounted as volumes so all outputs are persisted on your host machine.

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
