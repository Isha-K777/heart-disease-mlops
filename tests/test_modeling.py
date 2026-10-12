import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, train_test_split

from src.config import FEATURE_COLUMNS, TARGET
from src.modeling import (
    build_pipeline, evaluate_holdout, model_zoo, nested_cv, plot_importance, plot_roc,
)


def test_zoo_has_at_least_two_models():
    assert len(model_zoo()) >= 2


def test_pipeline_fit_predict_proba(clean_df):
    X, y = clean_df[FEATURE_COLUMNS], clean_df[TARGET]
    pipe = build_pipeline(LogisticRegression(max_iter=500)).fit(X, y)
    proba = pipe.predict_proba(X)
    assert proba.shape == (len(X), 2)
    assert np.allclose(proba.sum(axis=1), 1.0)


def test_pipeline_handles_missing_at_inference(clean_df):
    X, y = clean_df[FEATURE_COLUMNS], clean_df[TARGET]
    pipe = build_pipeline(LogisticRegression(max_iter=500)).fit(X, y)
    row = X.head(1).copy()
    row[["ca", "thal", "chol"]] = np.nan
    assert pipe.predict_proba(row).shape == (1, 2)


def test_evaluate_holdout_keys_and_ranges(clean_df):
    X, y = clean_df[FEATURE_COLUMNS], clean_df[TARGET]
    Xtr, Xte, ytr, yte = train_test_split(X, y, stratify=y, random_state=1)
    model = build_pipeline(LogisticRegression(max_iter=500)).fit(Xtr, ytr)
    metrics, proba = evaluate_holdout(model, Xte, yte)
    assert set(metrics) == {"accuracy", "precision", "recall", "f1", "roc_auc"}
    assert all(0.0 <= v <= 1.0 for v in metrics.values())
    assert len(proba) == len(Xte)


def test_nested_cv_returns_mean_and_std(clean_df):
    X, y = clean_df[FEATURE_COLUMNS], clean_df[TARGET]
    search = GridSearchCV(build_pipeline(LogisticRegression(max_iter=500)),
                          {"clf__C": [0.1, 1.0]}, cv=3, scoring="roc_auc")
    res = nested_cv(search, X, y, n_splits=3)
    assert set(res["roc_auc"]) == {"mean", "std"}
    assert 0.0 <= res["roc_auc"]["mean"] <= 1.0


def test_plots_render(clean_df):
    X, y = clean_df[FEATURE_COLUMNS], clean_df[TARGET]
    pipe = build_pipeline(LogisticRegression(max_iter=500)).fit(X, y)
    assert plot_importance(pipe) is not None
    assert plot_roc({"lr": (y, pipe.predict_proba(X)[:, 1])}) is not None
