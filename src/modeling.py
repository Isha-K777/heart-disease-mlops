"""Model zoo, evaluation helpers and plots (no MLflow dependency here)."""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.metrics import (  # noqa: E402
    ConfusionMatrixDisplay, accuracy_score, confusion_matrix, f1_score,
    precision_score, recall_score, roc_auc_score, roc_curve,
)
from sklearn.model_selection import StratifiedKFold, cross_validate  # noqa: E402
from sklearn.pipeline import Pipeline  # noqa: E402

from src.config import RANDOM_STATE  # noqa: E402
from src.features import ClinicalFeatures, build_preprocessor  # noqa: E402

SCORING = {
    "accuracy": "accuracy",
    "precision": "precision",
    "recall": "recall",
    "f1": "f1",
    "roc_auc": "roc_auc",
}


def build_pipeline(estimator) -> Pipeline:
    """Full reproducible pipeline: features -> preprocessing -> classifier."""
    return Pipeline([
        ("clinical", ClinicalFeatures()),
        ("prep", build_preprocessor()),
        ("clf", estimator),
    ])


def model_zoo() -> dict:
    """name -> (estimator, hyper-parameter grid). Grid keys target the 'clf' step."""
    return {
        "logistic_regression": (
            LogisticRegression(max_iter=2000, random_state=RANDOM_STATE),
            {"clf__C": [0.01, 0.1, 1.0, 10.0], "clf__class_weight": [None, "balanced"]},
        ),
        "random_forest": (
            RandomForestClassifier(random_state=RANDOM_STATE, n_jobs=1),
            {
                "clf__n_estimators": [200, 400],
                "clf__max_depth": [None, 5, 10],
                "clf__min_samples_leaf": [1, 3, 5],
            },
        ),
        "gradient_boosting": (
            GradientBoostingClassifier(random_state=RANDOM_STATE),
            {
                "clf__n_estimators": [100, 200],
                "clf__learning_rate": [0.03, 0.1],
                "clf__max_depth": [2, 3],
            },
        ),
    }


def nested_cv(search, X, y, n_splits: int = 5, seed: int = RANDOM_STATE) -> dict:
    """Nested CV: the outer loop scores a GridSearchCV (tuning happens inside each
    outer training fold), giving an unbiased estimate of tuned-model performance."""
    outer = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    res = cross_validate(search, X, y, cv=outer, scoring=SCORING, n_jobs=1)
    return {
        m: {"mean": float(np.mean(res[f"test_{m}"])), "std": float(np.std(res[f"test_{m}"]))}
        for m in SCORING
    }


def evaluate_holdout(model, X, y, threshold: float = 0.5):
    """Metrics on an unseen test set. Returns (metrics_dict, y_proba)."""
    proba = model.predict_proba(X)[:, 1]
    pred = (proba >= threshold).astype(int)
    metrics = {
        "accuracy": accuracy_score(y, pred),
        "precision": precision_score(y, pred, zero_division=0),
        "recall": recall_score(y, pred, zero_division=0),
        "f1": f1_score(y, pred, zero_division=0),
        "roc_auc": roc_auc_score(y, proba),
    }
    return {k: float(v) for k, v in metrics.items()}, proba


# ----------------------------------------------------------------- plots
def plot_roc(curves: dict):
    """curves: name -> (y_true, y_proba)."""
    fig, ax = plt.subplots(figsize=(6, 5))
    for name, (y_true, y_proba) in curves.items():
        fpr, tpr, _ = roc_curve(y_true, y_proba)
        ax.plot(fpr, tpr, lw=2, label=f"{name} (AUC={roc_auc_score(y_true, y_proba):.3f})")
    ax.plot([0, 1], [0, 1], "k--", lw=1, label="chance")
    ax.set(xlabel="False positive rate", ylabel="True positive rate",
           title="ROC curves (held-out test set)")
    ax.legend(loc="lower right")
    fig.tight_layout()
    return fig


def plot_confusion(y_true, y_pred, title: str):
    fig, ax = plt.subplots(figsize=(4.5, 4))
    cm = confusion_matrix(y_true, y_pred)
    ConfusionMatrixDisplay(cm, display_labels=["No disease", "Disease"]).plot(
        ax=ax, cmap="Blues", colorbar=False)
    ax.set_title(title)
    fig.tight_layout()
    return fig


def plot_importance(pipeline: Pipeline, top: int = 15):
    clf = pipeline.named_steps["clf"]
    names = pipeline.named_steps["prep"].get_feature_names_out()
    if hasattr(clf, "feature_importances_"):
        values, label = clf.feature_importances_, "Feature importance"
    elif hasattr(clf, "coef_"):
        values, label = np.abs(clf.coef_.ravel()), "|Coefficient| (standardised)"
    else:
        return None
    order = np.argsort(values)[::-1][:top][::-1]
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.barh([names[i] for i in order], values[order], color="#3b6ea5")
    ax.set(title="Top features", xlabel=label)
    fig.tight_layout()
    return fig
