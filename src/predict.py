"""Batch / CLI inference using the saved pipeline.

Usage: python -m src.predict --json sample_input.json
       python -m src.predict --csv data/processed/heart_clean.csv
"""
import argparse
import json

import joblib
import pandas as pd

from src.config import FEATURE_COLUMNS, MODEL_PATH


def load_model(path=MODEL_PATH):
    return joblib.load(path)


def predict_frame(model, df: pd.DataFrame) -> pd.DataFrame:
    X = df.reindex(columns=FEATURE_COLUMNS).astype("float64")
    proba = model.predict_proba(X)[:, 1]
    out = df.copy()
    out["probability_disease"] = proba
    out["prediction"] = (proba >= 0.5).astype(int)
    out["confidence"] = pd.Series(proba).where(proba >= 0.5, 1 - proba).values
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--json")
    ap.add_argument("--csv")
    a = ap.parse_args()
    if a.json:
        with open(a.json) as fh:
            data = json.load(fh)
        frame = pd.DataFrame([data] if isinstance(data, dict) else data)
    elif a.csv:
        frame = pd.read_csv(a.csv)
    else:
        ap.error("provide --json or --csv")
    print(predict_frame(load_model(), frame)[
        ["prediction", "confidence", "probability_disease"]].to_string())
