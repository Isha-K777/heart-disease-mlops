"""Download the UCI Heart Disease (Cleveland) dataset and build the cleaned CSV.

Usage: python scripts/download_data.py [--force]

Source: https://archive.ics.uci.edu/dataset/45/heart+disease
Fallback if the URL is blocked: `pip install ucimlrepo` (used automatically),
or download processed.cleveland.data manually into data/raw/ and re-run.
"""
import argparse
import io
import logging
import sys
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd  # noqa: E402

from src.config import DATA_CLEAN, DATA_RAW, RAW_COLUMNS  # noqa: E402
from src.data import clean, save_clean  # noqa: E402

URL = ("https://archive.ics.uci.edu/ml/machine-learning-databases/"
       "heart-disease/processed.cleveland.data")
log = logging.getLogger("download")


def fetch_direct() -> pd.DataFrame:
    with urllib.request.urlopen(URL, timeout=30) as resp:
        text = resp.read().decode("utf-8")
    return pd.read_csv(io.StringIO(text), header=None, names=RAW_COLUMNS, na_values="?")


def fetch_ucimlrepo() -> pd.DataFrame:
    from ucimlrepo import fetch_ucirepo  # optional dependency

    ds = fetch_ucirepo(id=45)
    df = pd.concat([ds.data.features, ds.data.targets], axis=1)
    df.columns = RAW_COLUMNS
    return df


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true", help="re-download even if file exists")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    if DATA_RAW.exists() and not args.force:
        log.info("Raw data already present at %s (use --force to refresh)", DATA_RAW)
        raw = pd.read_csv(DATA_RAW, na_values="?")
    else:
        try:
            raw = fetch_direct()
            log.info("Downloaded from UCI archive")
        except Exception as exc:  # network / URL problems
            log.warning("Direct download failed (%s); trying ucimlrepo", exc)
            raw = fetch_ucimlrepo()
        DATA_RAW.parent.mkdir(parents=True, exist_ok=True)
        raw.to_csv(DATA_RAW, index=False)
        log.info("Saved raw data -> %s (%d rows)", DATA_RAW, len(raw))

    cleaned = clean(raw)
    save_clean(cleaned, DATA_CLEAN)
    log.info("Saved cleaned data -> %s | shape=%s | positives=%.1f%%",
             DATA_CLEAN, cleaned.shape, 100 * cleaned["target"].mean())


if __name__ == "__main__":
    main()
