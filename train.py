"""
train.py — model training, evaluation, and comparison for bank term-deposit prediction.

Three models are trained and evaluated:
  1. Logistic Regression
  2. Decision Tree
  3. Random Forest

All use class_weight="balanced" to address the ~88/12% class imbalance.

Evaluation metrics reported per model:
  - Accuracy, Precision, Recall, F1-score (macro and weighted), ROC-AUC
  - Confusion matrix (saved as PNG)
  - Combined ROC curve (saved as PNG)

Model selection is NOT automatic. The notebook prose explains the choice
after inspecting the full results table with the project objective in mind.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import FIGURES_DIR, RANDOM_STATE, TEST_SIZE  # noqa: E402
from src.preprocessing import CATEGORICAL_FEATURES, NUMERIC_FEATURES, build_preprocessor, get_X_y  # noqa: E402

sns.set_theme(style="whitegrid")
plt.rcParams.update({"figure.dpi": 120, "savefig.bbox": "tight"})


# ── Model definitions ─────────────────────────────────────────────────────────

def get_model_definitions() -> dict[str, Any]:
    """Return the three classifier definitions used in this project."""
    return {
        "Logistic Regression": LogisticRegression(
            class_weight="balanced",
            max_iter=1000,
            random_state=RANDOM_STATE,
            solver="lbfgs",
        ),
        "Decision Tree": DecisionTreeClassifier(
            class_weight="balanced",
            random_state=RANDOM_STATE,
        ),
        "Random Forest": RandomForestClassifier(
            class_weight="balanced",
            n_estimators=100,
            random_state=RANDOM_STATE,
            n_jobs=-1,
        ),
    }


# ── Split ─────────────────────────────────────────────────────────────────────

def split_data(
    df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Stratified 80/20 train/test split.

    Stratification ensures the minority class (~11.7%) is proportionally
    represented in both train and test sets.
    """
    X, y = get_X_y(df)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y,
    )
    print(
        f"Train: {len(X_train):,} rows  |  "
        f"Test: {len(X_test):,} rows  |  "
        f"Positive rate — train: {y_train.mean():.3f}  test: {y_test.mean():.3f}"
    )
    return X_train, X_test, y_train, y_test


# ── Build pipeline ────────────────────────────────────────────────────────────

def build_pipeline(estimator: Any) -> Pipeline:
    """Wrap a preprocessor + estimator into a single sklearn Pipeline."""
    return Pipeline(
        steps=[
            ("preprocessor", build_preprocessor()),
            ("classifier", estimator),
        ]
    )


# ── Evaluate a single model ───────────────────────────────────────────────────

def evaluate_model(
    name: str,
    pipeline: Pipeline,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    save_cm: bool = True,
) -> dict[str, float]:
    """Compute and return evaluation metrics for a fitted pipeline.

    Parameters
    ----------
    name : str
        Human-readable model name (used in plot titles and filenames).
    pipeline : fitted sklearn Pipeline
    X_test, y_test : held-out test data
    save_cm : bool
        If True, saves the confusion matrix to outputs/figures/.

    Returns
    -------
    dict with keys: accuracy, precision, recall, f1, roc_auc
    """
    y_pred = pipeline.predict(X_test)
    y_prob = pipeline.predict_proba(X_test)[:, 1]

    metrics = {
        "accuracy":  accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, zero_division=0),
        "recall":    recall_score(y_test, y_pred, zero_division=0),
        "f1":        f1_score(y_test, y_pred, zero_division=0),
        "roc_auc":   roc_auc_score(y_test, y_prob),
    }

    print(f"\n{'=' * 50}")
    print(f"  {name}")
    print(f"{'=' * 50}")
    for k, v in metrics.items():
        print(f"  {k:<12}: {v:.4f}")
    print()
    print(classification_report(y_test, y_pred, target_names=["No", "Yes"], zero_division=0))

    if save_cm:
        _plot_confusion_matrix(name, y_test, y_pred)

    return metrics


def _plot_confusion_matrix(name: str, y_test: pd.Series, y_pred: np.ndarray) -> None:
    """Save a confusion matrix plot for the given model."""
    cm = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots(figsize=(5, 4))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["No", "Yes"])
    disp.plot(ax=ax, colorbar=False, cmap="Blues")
    ax.set_title(f"Confusion Matrix — {name}", fontsize=12, fontweight="bold")
    fig.tight_layout()
    fname = name.lower().replace(" ", "_")
    path = FIGURES_DIR / f"cm_{fname}.png"
    fig.savefig(path)
    plt.close(fig)
    print(f"Saved confusion matrix: {path.name}")


# ── Train all models ──────────────────────────────────────────────────────────

def train_all_models(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
) -> tuple[dict[str, Pipeline], pd.DataFrame]:
    """Train all three models and return fitted pipelines + metrics comparison table.

    Returns
    -------
    pipelines : dict[model_name -> fitted Pipeline]
    results_df : DataFrame with rows = models, cols = metrics
    """
    model_defs = get_model_definitions()
    pipelines: dict[str, Pipeline] = {}
    results: dict[str, dict] = {}

    for name, estimator in model_defs.items():
        print(f"\nTraining {name}...")
        pipeline = build_pipeline(estimator)
        pipeline.fit(X_train, y_train)
        pipelines[name] = pipeline
        metrics = evaluate_model(name, pipeline, X_test, y_test)
        results[name] = metrics

    results_df = pd.DataFrame(results).T
    results_df.index.name = "Model"
    results_df = results_df[["accuracy", "precision", "recall", "f1", "roc_auc"]]
    results_df.columns = ["Accuracy", "Precision", "Recall", "F1", "ROC-AUC"]

    return pipelines, results_df


# ── ROC curve comparison ──────────────────────────────────────────────────────

def plot_roc_curves(
    pipelines: dict[str, Pipeline],
    X_test: pd.DataFrame,
    y_test: pd.Series,
) -> plt.Figure:
    """Plot ROC curves for all models on a single chart."""
    fig, ax = plt.subplots(figsize=(8, 6))
    colors = ["#4878CF", "#e05353", "#56a14b"]

    for (name, pipeline), color in zip(pipelines.items(), colors):
        y_prob = pipeline.predict_proba(X_test)[:, 1]
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        auc = roc_auc_score(y_test, y_prob)
        ax.plot(fpr, tpr, label=f"{name}  (AUC = {auc:.3f})", color=color, linewidth=2)

    ax.plot([0, 1], [0, 1], "k--", linewidth=1, label="Random classifier")
    ax.set_xlabel("False Positive Rate", fontsize=12)
    ax.set_ylabel("True Positive Rate", fontsize=12)
    ax.set_title("ROC Curves — Model Comparison", fontsize=13, fontweight="bold")
    ax.legend(loc="lower right", fontsize=10)
    ax.grid(True, alpha=0.4)
    fig.tight_layout()

    path = FIGURES_DIR / "roc_curves.png"
    fig.savefig(path)
    print(f"Saved ROC curves: {path.name}")
    return fig


# ── Metrics comparison table ──────────────────────────────────────────────────

def plot_metrics_comparison(results_df: pd.DataFrame) -> plt.Figure:
    """Bar chart comparing all metrics across models."""
    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(len(results_df))
    metrics = results_df.columns.tolist()
    width = 0.15
    colors = ["#4878CF", "#e05353", "#56a14b", "#e8a838", "#9b59b6"]

    for i, (metric, color) in enumerate(zip(metrics, colors)):
        bars = ax.bar(x + i * width, results_df[metric], width, label=metric, color=color, edgecolor="white")

    ax.set_xticks(x + width * (len(metrics) - 1) / 2)
    ax.set_xticklabels(results_df.index, fontsize=11)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Score")
    ax.set_title("Model Comparison — All Metrics", fontsize=13, fontweight="bold")
    ax.legend(loc="lower right", fontsize=9)
    ax.grid(axis="y", alpha=0.4)
    fig.tight_layout()

    path = FIGURES_DIR / "metrics_comparison.png"
    fig.savefig(path)
    print(f"Saved metrics comparison: {path.name}")
    return fig


# ── Standalone runner ─────────────────────────────────────────────────────────

if __name__ == "__main__":
    from src.data_loader import load_data

    df = load_data()
    X_train, X_test, y_train, y_test = split_data(df)
    pipelines, results_df = train_all_models(X_train, y_train, X_test, y_test)

    print("\n\n=== METRICS COMPARISON TABLE ===")
    print(results_df.to_string(float_format="{:.4f}".format))

    plot_roc_curves(pipelines, X_test, y_test)
    plot_metrics_comparison(results_df)
    plt.close("all")
