"""Train, tune, evaluate and track all candidate models with MLflow.

Run:  python -m src.train
View: mlflow ui --backend-store-uri ./mlruns
"""
import argparse
import json
import logging
import os
import platform
from datetime import datetime, timezone

import joblib
import mlflow
import mlflow.sklearn
import pandas as pd
import sklearn
from mlflow.models import infer_signature
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split

from src import config
from src.data import load_clean, split_xy
from src.modeling import (
    build_pipeline, evaluate_holdout, model_zoo, nested_cv,
    plot_confusion, plot_importance, plot_roc,
)

logger = logging.getLogger("train")


def _save_fig(fig, path):
    if fig is None:
        return None
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    import matplotlib.pyplot as plt
    plt.close(fig)
    return path


def run(args) -> dict:
    df = load_clean()
    X, y = split_xy(df)
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=args.test_size, stratify=y, random_state=config.RANDOM_STATE)
    logger.info("Train=%d Test=%d positive-rate=%.3f", len(X_tr), len(X_te), y.mean())

    tracking_uri = os.getenv("MLFLOW_TRACKING_URI", f"file:{config.ROOT / 'mlruns'}")
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(args.experiment)

    fig_dir = config.FIGURES_DIR / "models"
    inner_cv = StratifiedKFold(args.inner_folds, shuffle=True, random_state=config.RANDOM_STATE)
    results, fitted, roc_inputs = {}, {}, {}

    for name, (estimator, grid) in model_zoo().items():
        if args.models and name not in args.models:
            continue
        with mlflow.start_run(run_name=name) as run:
            logger.info("=== %s ===", name)
            search = GridSearchCV(build_pipeline(estimator), grid, cv=inner_cv,
                                  scoring="roc_auc", n_jobs=1, refit=True)

            cv = nested_cv(search, X_tr, y_tr, n_splits=args.outer_folds)
            search.fit(X_tr, y_tr)
            best = search.best_estimator_
            test_metrics, proba = evaluate_holdout(best, X_te, y_te)

            n_combos = len(search.cv_results_["params"])
            mlflow.set_tags({"model_family": name, "stage": "experiment"})
            mlflow.log_params({
                "model_name": name, "test_size": args.test_size,
                "inner_cv_folds": args.inner_folds, "outer_cv_folds": args.outer_folds,
                "random_state": config.RANDOM_STATE, "n_train": len(X_tr),
                "n_test": len(X_te), "grid_combinations": n_combos,
                "tuning_metric": "roc_auc", "sklearn_version": sklearn.__version__,
                **{f"best_{k}": v for k, v in search.best_params_.items()},
            })
            metrics = {"inner_best_roc_auc": float(search.best_score_)}
            for m, s in cv.items():
                metrics[f"cv_{m}_mean"] = s["mean"]
                metrics[f"cv_{m}_std"] = s["std"]
            for m, v in test_metrics.items():
                metrics[f"test_{m}"] = v
            mlflow.log_metrics(metrics)

            # artifacts & plots
            cvr = pd.DataFrame(search.cv_results_)[
                ["params", "mean_test_score", "std_test_score", "rank_test_score"]]
            cvr_path = fig_dir / f"{name}_grid_search.csv"
            cvr_path.parent.mkdir(parents=True, exist_ok=True)
            cvr.sort_values("rank_test_score").to_csv(cvr_path, index=False)
            artifacts = [
                cvr_path,
                _save_fig(plot_roc({name: (y_te, proba)}), fig_dir / f"{name}_roc.png"),
                _save_fig(plot_confusion(y_te, (proba >= 0.5).astype(int), name),
                          fig_dir / f"{name}_confusion.png"),
                _save_fig(plot_importance(best), fig_dir / f"{name}_importance.png"),
            ]
            for a in artifacts:
                if a is not None:
                    mlflow.log_artifact(str(a), artifact_path="plots")

            signature = infer_signature(X_tr.head(5), best.predict_proba(X_tr.head(5)))
            mlflow.sklearn.log_model(best, artifact_path="model", signature=signature)

            results[name] = {"run_id": run.info.run_id, "cv": cv, "test": test_metrics,
                             "best_params": {k: (v if v is not None else "None")
                                             for k, v in search.best_params_.items()}}
            fitted[name] = best
            roc_inputs[name] = (y_te, proba)
            logger.info("%s | nested-CV AUC %.3f | test AUC %.3f", name,
                        cv["roc_auc"]["mean"], test_metrics["roc_auc"])

    # ------------------------------------------------ model selection
    best_name = max(results, key=lambda n: results[n]["cv"]["roc_auc"]["mean"])
    logger.info("Selected model: %s", best_name)
    config.MODELS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(fitted[best_name], config.MODEL_PATH)

    info = {
        "model_name": best_name,
        "version": datetime.now(timezone.utc).strftime("%Y%m%d%H%M%S"),
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "selection_metric": "nested-CV ROC-AUC",
        "sklearn_version": sklearn.__version__,
        "python_version": platform.python_version(),
        "features": config.FEATURE_COLUMNS,
        "mlflow_run_id": results[best_name]["run_id"],
        "results": results,
    }
    config.MODEL_INFO_PATH.write_text(json.dumps(info, indent=2, default=str))

    # comparison table for the report
    rows = ["| Model | CV Acc | CV Prec | CV Rec | CV F1 | CV ROC-AUC | Test ROC-AUC |",
            "|---|---|---|---|---|---|---|"]
    for n, r in results.items():
        c = r["cv"]
        rows.append(
            f"| {n}{' **(selected)**' if n == best_name else ''} "
            + " ".join(f"| {c[m]['mean']:.3f} ± {c[m]['std']:.3f} "
                       for m in ("accuracy", "precision", "recall", "f1", "roc_auc"))
            + f"| {r['test']['roc_auc']:.3f} |")
    config.REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    (config.REPORTS_DIR / "model_comparison.md").write_text("\n".join(rows) + "\n")

    with mlflow.start_run(run_name="model_selection"):
        mlflow.set_tags({"stage": "selection", "selected_model": best_name})
        mlflow.log_param("selected_model", best_name)
        mlflow.log_metric("selected_cv_roc_auc", results[best_name]["cv"]["roc_auc"]["mean"])
        mlflow.log_metric("selected_test_roc_auc", results[best_name]["test"]["roc_auc"])
        cmp_fig = _save_fig(plot_roc(roc_inputs), fig_dir / "all_models_roc.png")
        mlflow.log_artifact(str(cmp_fig), artifact_path="plots")
        mlflow.log_artifact(str(config.REPORTS_DIR / "model_comparison.md"))
        mlflow.log_artifact(str(config.MODEL_INFO_PATH))
        mlflow.log_artifact(str(config.ROOT / "requirements.txt"))

    print("\n" + (config.REPORTS_DIR / "model_comparison.md").read_text())
    return info


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--test-size", type=float, default=0.2)
    p.add_argument("--inner-folds", type=int, default=3)
    p.add_argument("--outer-folds", type=int, default=5)
    p.add_argument("--experiment", default=config.MLFLOW_EXPERIMENT)
    p.add_argument("--models", nargs="*", help="subset, e.g. logistic_regression")
    return p.parse_args(argv)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    run(parse_args())
