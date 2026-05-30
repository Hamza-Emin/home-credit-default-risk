import hashlib
from pathlib import Path

import numpy as np
import pandas as pd
from omegaconf import DictConfig
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder


def _encode_categoricals(
    df: pd.DataFrame,
) -> tuple[pd.DataFrame, list[str], dict]:
    cat_cols = df.select_dtypes(include="object").columns.tolist()
    encoders: dict = {}
    for col in cat_cols:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))
        encoders[col] = le
    return df, cat_cols, encoders


def _get_or_create_splits(
    n_samples: int,
    labels: np.ndarray,
    cfg: DictConfig,
    data_path: Path,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    split_file = data_path / "split_indices.npz"
    hash_file = data_path / "split_indices_hash.txt"

    if split_file.exists():
        splits = np.load(split_file)
        return splits["train_idx"], splits["val_idx"], splits["test_idx"]

    indices = np.arange(n_samples)

    # 70 / 15 / 15 stratified split
    train_val_idx, test_idx = train_test_split(
        indices,
        test_size=0.15,
        stratify=labels,
        random_state=cfg.data.random_state,
    )
    train_idx, val_idx = train_test_split(
        train_val_idx,
        test_size=0.15 / 0.85,  # 15% of total from the remaining 85%
        stratify=labels[train_val_idx],
        random_state=cfg.data.random_state,
    )

    np.savez(split_file, train_idx=train_idx, val_idx=val_idx, test_idx=test_idx)

    file_hash = hashlib.sha256(split_file.read_bytes()).hexdigest()
    hash_file.write_text(file_hash)

    print(f"Splits created and saved → {split_file}")
    print(f"  SHA256: {file_hash}")
    return train_idx, val_idx, test_idx


def _apply_smote(
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    cat_cols: list[str],
    random_state: int,
) -> tuple[pd.DataFrame, np.ndarray]:
    from imblearn.over_sampling import SMOTENC

    cat_indices = [X_train.columns.get_loc(col) for col in cat_cols]
    sampler = SMOTENC(categorical_features=cat_indices, random_state=random_state)
    X_resampled, y_resampled = sampler.fit_resample(X_train, y_train)
    print(
        f"SMOTE: {len(y_train)} → {len(y_resampled)} samples "
        f"| class ratio: {(y_resampled == 0).sum()}/{(y_resampled == 1).sum()}"
    )
    return pd.DataFrame(X_resampled, columns=X_train.columns), y_resampled


def load_data(cfg: DictConfig) -> dict:
    data_path = Path(cfg.data.data_path)
    merged_csv = data_path / "merged_train.csv"

    if not merged_csv.exists():
        raise FileNotFoundError(
            f"{merged_csv} not found. Run `uv run python main.py preprocess` first."
        )

    df = pd.read_csv(merged_csv)
    df = df.replace([float("inf"), float("-inf")], float("nan"))
    num_cols = df.select_dtypes(include="number").columns
    df[num_cols] = df[num_cols].fillna(0)

    df, cat_cols, encoders = _encode_categoricals(df)

    target_col = cfg.data.target_col
    id_col = cfg.data.id_col
    feature_cols = [c for c in df.columns if c not in (target_col, id_col)]

    features = df[feature_cols].copy()
    labels = df[target_col].values
    cont_cols = [c for c in feature_cols if c not in cat_cols]

    train_idx, val_idx, test_idx = _get_or_create_splits(
        len(features), labels, cfg, data_path
    )

    X_train = features.iloc[train_idx].reset_index(drop=True)
    y_train = labels[train_idx]

    # SMOTE applies only to training data — val/test stay untouched
    X_train, y_train = _apply_smote(X_train, y_train, cat_cols, cfg.data.random_state)

    return {
        "X_train": X_train,
        "X_val": features.iloc[val_idx].reset_index(drop=True),
        "X_test": features.iloc[test_idx].reset_index(drop=True),
        "y_train": y_train,
        "y_val": labels[val_idx],
        "y_test": labels[test_idx],
        "cat_cols": cat_cols,
        "cont_cols": cont_cols,
        "encoders": encoders,
    }
