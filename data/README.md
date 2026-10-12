# Data
Generate both files (and commit them) with: `python scripts/download_data.py`

* `raw/heart_disease_raw.csv` - UCI Cleveland data with header ('?' = missing)
* `processed/heart_clean.csv` - cleaned (binary `target`, float features, NaNs kept for in-pipeline imputation)

Source: https://archive.ics.uci.edu/dataset/45/heart+disease
