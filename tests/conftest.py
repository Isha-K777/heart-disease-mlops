"""Shared fixtures. Tests use synthetic data so they run offline in CI."""
import joblib
import numpy as np
import pandas as pd
import pytest

from src.config import RAW_COLUMNS
from src.data import clean
from src.modeling import build_pipeline
from sklearn.linear_model import LogisticRegression


def make_raw(n: int = 240, seed: int = 0) -> pd.DataFrame:
    """Synthetic frame mimicking the UCI raw schema (incl. '?' markers)."""
    rng = np.random.default_rng(seed)
    age = rng.integers(30, 78, n)
    thalach = rng.integers(95, 200, n)
    oldpeak = np.round(rng.gamma(1.2, 0.9, n), 1)
    cp = rng.integers(1, 5, n)
    exang = rng.integers(0, 2, n)
    logit = 0.04 * (age - 54) - 0.03 * (thalach - 150) + 0.6 * oldpeak + 0.5 * exang \
        + 0.4 * (cp == 4) - 0.3
    num = (rng.random(n) < 1 / (1 + np.exp(-logit))).astype(int) * rng.integers(1, 5, n)
    df = pd.DataFrame({
        "age": age, "sex": rng.integers(0, 2, n), "cp": cp,
        "trestbps": rng.integers(95, 200, n), "chol": rng.integers(125, 400, n),
        "fbs": rng.integers(0, 2, n), "restecg": rng.integers(0, 3, n),
        "thalach": thalach, "exang": exang, "oldpeak": oldpeak,
        "slope": rng.integers(1, 4, n), "ca": rng.integers(0, 4, n).astype(object),
        "thal": rng.choice([3, 6, 7], n).astype(object), "num": num,
    })[RAW_COLUMNS]
    df.loc[df.sample(4, random_state=seed).index, "ca"] = "?"
    df.loc[df.sample(2, random_state=seed + 1).index, "thal"] = "?"
    return df


@pytest.fixture(scope="session")
def raw_df():
    return make_raw()


@pytest.fixture(scope="session")
def clean_df(raw_df):
    return clean(raw_df)


@pytest.fixture(scope="session")
def model_path(tmp_path_factory, clean_df):
    from src.config import FEATURE_COLUMNS, TARGET
    model = build_pipeline(LogisticRegression(max_iter=500)).fit(
        clean_df[FEATURE_COLUMNS], clean_df[TARGET])
    path = tmp_path_factory.mktemp("model") / "model.joblib"
    joblib.dump(model, path)
    return path
