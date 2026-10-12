"""Data loading and cleaning.

Design note: missing values are *kept* as NaN in the cleaned CSV and imputed
inside the sklearn pipeline. Imputing here (before the train/test split) would
leak test-set statistics into training.
"""
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from src.config import DATA_CLEAN, DATA_RAW, FEATURE_COLUMNS, RAW_COLUMNS, TARGET

logger = logging.getLogger(__name__)


def load_raw(path: Path = DATA_RAW) -> pd.DataFrame:
    """Read the raw CSV produced by scripts/download_data.py ('?' = missing)."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Run `python scripts/download_data.py` first."
        )
    return pd.read_csv(path, na_values=["?", ""])


def clean(df: pd.DataFrame) -> pd.DataFrame:
    """Return an analysis-ready frame.

    Steps: validate columns, coerce numerics, binarise target (num > 0),
    drop exact duplicates, cast features to float64.
    """
    missing_cols = [c for c in RAW_COLUMNS if c not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing expected columns: {missing_cols}")

    out = df[RAW_COLUMNS].copy()
    out = out.replace("?", np.nan).infer_objects(copy=False)
    for col in RAW_COLUMNS:
        out[col] = pd.to_numeric(out[col], errors="coerce")

    if out["num"].isna().any():
        logger.warning("Dropping %d rows with missing target", out["num"].isna().sum())
        out = out.dropna(subset=["num"])

    out[TARGET] = (out["num"] > 0).astype(int)
    out = out.drop(columns=["num"])

    before = len(out)
    out = out.drop_duplicates().reset_index(drop=True)
    if len(out) < before:
        logger.info("Dropped %d duplicate rows", before - len(out))

    out[FEATURE_COLUMNS] = out[FEATURE_COLUMNS].astype("float64")
    return out


def save_clean(df: pd.DataFrame, path: Path = DATA_CLEAN) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    return path


def load_clean(path: Path = DATA_CLEAN) -> pd.DataFrame:
    """Load the cleaned dataset, building it from raw data if necessary."""
    path = Path(path)
    if not path.exists():
        logger.info("Cleaned data not found; building from raw data")
        save_clean(clean(load_raw()), path)
    return pd.read_csv(path)


def split_xy(df: pd.DataFrame):
    return df[FEATURE_COLUMNS], df[TARGET]
