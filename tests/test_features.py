import numpy as np

from src.config import ENGINEERED_FEATURES, FEATURE_COLUMNS
from src.features import ClinicalFeatures, build_preprocessor


def test_engineered_columns_added(clean_df):
    out = ClinicalFeatures().fit_transform(clean_df[FEATURE_COLUMNS])
    for col in ENGINEERED_FEATURES:
        assert col in out.columns
    assert np.isfinite(out[ENGINEERED_FEATURES]).all().all()


def test_hr_reserve_formula(clean_df):
    X = clean_df[FEATURE_COLUMNS].head(5)
    out = ClinicalFeatures().transform(X)
    expected = X["thalach"] / (220 - X["age"])
    np.testing.assert_allclose(out["hr_reserve_pct"], expected)


def test_preprocessor_has_no_nans_and_scales(clean_df):
    X = ClinicalFeatures().transform(clean_df[FEATURE_COLUMNS])
    Z = build_preprocessor().fit_transform(X)
    assert not np.isnan(Z).any()
    assert Z.shape[0] == len(X)


def test_unseen_category_is_ignored(clean_df):
    X = ClinicalFeatures().transform(clean_df[FEATURE_COLUMNS])
    prep = build_preprocessor().fit(X)
    X_new = X.head(2).copy()
    X_new["cp"] = 99.0
    assert prep.transform(X_new).shape[0] == 2
