"""Exploratory data analysis: figures + a JSON summary."""
import json
import logging

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import seaborn as sns  # noqa: E402

from src.config import (  # noqa: E402
    CATEGORICAL_FEATURES, FIGURES_DIR, NUMERIC_FEATURES, REPORTS_DIR, TARGET,
)

logger = logging.getLogger(__name__)
PALETTE = {"No disease": "#4c9f70", "Disease": "#d1495b"}
sns.set_theme(style="whitegrid", context="notebook")


def _label(df):
    return df.assign(diagnosis=df[TARGET].map({0: "No disease", 1: "Disease"}))


def _save(fig, name):
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    path = FIGURES_DIR / name
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_class_balance(df):
    d = _label(df)
    fig, ax = plt.subplots(figsize=(5, 4))
    sns.countplot(data=d, x="diagnosis", hue="diagnosis", palette=PALETTE,
                  order=["No disease", "Disease"], legend=False, ax=ax)
    total = len(d)
    for p in ax.patches:
        ax.annotate(f"{int(p.get_height())} ({p.get_height() / total:.1%})",
                    (p.get_x() + p.get_width() / 2, p.get_height()),
                    ha="center", va="bottom")
    ax.set(title="Class balance", xlabel="", ylabel="Patients")
    return _save(fig, "eda_class_balance.png")


def plot_missing(df):
    miss = df.isna().sum()
    miss = miss[miss > 0]
    fig, ax = plt.subplots(figsize=(5, 3.5))
    if miss.empty:
        ax.text(0.5, 0.5, "No missing values", ha="center", va="center")
        ax.axis("off")
    else:
        sns.barplot(x=miss.index, y=miss.values, color="#3b6ea5", ax=ax)
        ax.set(title="Missing values per column", ylabel="Count")
    return _save(fig, "eda_missing_values.png")


def plot_numeric_hists(df):
    d = _label(df)
    fig, axes = plt.subplots(2, 3, figsize=(14, 7))
    for ax, col in zip(axes.ravel(), NUMERIC_FEATURES):
        sns.histplot(data=d, x=col, hue="diagnosis", palette=PALETTE, kde=True,
                     bins=25, alpha=0.5, ax=ax)
        ax.set_title(f"Distribution of {col}")
    for ax in axes.ravel()[len(NUMERIC_FEATURES):]:
        ax.axis("off")
    fig.suptitle("Numeric feature distributions by diagnosis", y=1.02, fontsize=14)
    fig.tight_layout()
    return _save(fig, "eda_numeric_histograms.png")


def plot_numeric_boxplots(df):
    d = _label(df)
    fig, axes = plt.subplots(1, len(NUMERIC_FEATURES), figsize=(16, 4))
    for ax, col in zip(axes, NUMERIC_FEATURES):
        sns.boxplot(data=d, x="diagnosis", y=col, hue="diagnosis", palette=PALETTE,
                    order=["No disease", "Disease"], legend=False, ax=ax)
        ax.set_xlabel("")
    fig.suptitle("Outlier check: numeric features by diagnosis", y=1.03)
    fig.tight_layout()
    return _save(fig, "eda_numeric_boxplots.png")


def plot_categorical(df):
    d = _label(df)
    fig, axes = plt.subplots(2, 4, figsize=(16, 7))
    for ax, col in zip(axes.ravel(), CATEGORICAL_FEATURES):
        sns.countplot(data=d, x=col, hue="diagnosis", palette=PALETTE, ax=ax)
        ax.set_title(col)
        ax.legend(title="", fontsize=8)
    fig.suptitle("Categorical features by diagnosis", y=1.02, fontsize=14)
    fig.tight_layout()
    return _save(fig, "eda_categorical_counts.png")


def plot_correlation(df):
    corr = df.corr(numeric_only=True)
    mask = np.triu(np.ones_like(corr, dtype=bool), k=1)
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(corr, mask=mask, annot=True, fmt=".2f", cmap="coolwarm", center=0,
                square=True, linewidths=0.5, cbar_kws={"shrink": 0.8}, ax=ax)
    ax.set_title("Correlation heatmap (Pearson)")
    return _save(fig, "eda_correlation_heatmap.png")


def run_eda(df) -> dict:
    """Generate all figures and write reports/eda_summary.json."""
    paths = [
        plot_class_balance(df), plot_missing(df), plot_numeric_hists(df),
        plot_numeric_boxplots(df), plot_categorical(df), plot_correlation(df),
    ]
    corr_target = (df.corr(numeric_only=True)[TARGET].drop(TARGET)
                   .sort_values(key=abs, ascending=False))
    summary = {
        "rows": int(len(df)),
        "columns": int(df.shape[1]),
        "class_counts": {str(k): int(v) for k, v in df[TARGET].value_counts().items()},
        "positive_rate": float(df[TARGET].mean()),
        "missing_values": {k: int(v) for k, v in df.isna().sum().items() if v > 0},
        "top_correlations_with_target": {k: round(float(v), 3)
                                         for k, v in corr_target.head(6).items()},
        "figures": [str(p.name) for p in paths],
    }
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    (REPORTS_DIR / "eda_summary.json").write_text(json.dumps(summary, indent=2))
    logger.info("EDA complete: %s", summary)
    return summary


if __name__ == "__main__":
    from src.data import load_clean

    logging.basicConfig(level=logging.INFO)
    print(json.dumps(run_eda(load_clean()), indent=2))
