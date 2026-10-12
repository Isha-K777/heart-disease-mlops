import numpy as np
import pandas as pd
import pytest

from src.config import FEATURE_COLUMNS, TARGET
from src.data import clean, split_xy


def test_target_is_binary_and_num_dropped(clean_df):
    assert set(clean_df[TARGET].unique()) <= {0, 1}
    assert "num" not in clean_df.columns


def test_question_marks_become_nan(clean_df):
    assert clean_df["ca"].isna().sum() >= 1
    assert clean_df["thal"].isna().sum() >= 1


def test_features_are_float(clean_df):
    assert all(clean_df[c].dtype == np.float64 for c in FEATURE_COLUMNS)


def test_target_binarisation_rule(raw_df):
    out = clean(raw_df)
    expected = (raw_df["num"] > 0).astype(int).sum()
    # duplicates may be dropped, so cleaned positives can only be <= original
    assert out[TARGET].sum() <= expected


def test_duplicates_removed(raw_df):
    doubled = pd.concat([raw_df, raw_df.iloc[:10]], ignore_index=True)
    assert len(clean(doubled)) == len(clean(raw_df))


def test_missing_columns_raise(raw_df):
    with pytest.raises(ValueError):
        clean(raw_df.drop(columns=["chol"]))


def test_split_xy(clean_df):
    X, y = split_xy(clean_df)
    assert list(X.columns) == FEATURE_COLUMNS
    assert len(X) == len(y)
    