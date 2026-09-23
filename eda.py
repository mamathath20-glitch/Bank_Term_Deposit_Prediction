"""
eda.py — reusable EDA plot functions for the bank marketing dataset.

All functions:
 - Accept a DataFrame (as returned by data_loader.load_data)
 - Save their figure to FIGURES_DIR as a PNG
 - Return the matplotlib Figure object so the notebook can display it inline

Usage in the notebook:
    from src.eda import (
        plot_target_distribution,
        plot_numeric_distributions,
        plot_categorical_distributions,
        plot_correlation_heatmap,
        plot_pdays_analysis,
        plot_subscription_by_feature,
        plot_unknown_audit,
    )
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import numpy as np
import pandas as pd
import seaborn as sns

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import FIGURES_DIR  # noqa: E402

# ── Style ─────────────────────────────────────────────────────────────────────
sns.set_theme(style="whitegrid", palette="muted")
plt.rcParams.update({"figure.dpi": 120, "savefig.bbox": "tight"})


def _save(fig: plt.Figure, name: str) -> Path:
    path = FIGURES_DIR / f"{name}.png"
    fig.savefig(path)
    print(f"Saved: {path.relative_to(Path.cwd()) if Path.cwd() in path.parents else path}")
    return path


# ── 1. Target distribution ────────────────────────────────────────────────────

def plot_target_distribution(df: pd.DataFrame) -> plt.Figure:
    """Bar chart showing class imbalance in the target variable."""
    counts = df["y"].value_counts().rename({0: "No (did not subscribe)", 1: "Yes (subscribed)"})
    pcts = counts / counts.sum() * 100

    fig, ax = plt.subplots(figsize=(6, 4))
    bars = ax.bar(counts.index, counts.values, color=["#4878CF", "#e05353"], edgecolor="white")
    for bar, pct in zip(bars, pcts):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 200,
            f"{pct:.1f}%",
            ha="center", va="bottom", fontsize=11, fontweight="bold",
        )
    ax.set_title("Target Variable Distribution\n(Significant Class Imbalance)", fontsize=13, fontweight="bold")
    ax.set_ylabel("Number of Customers")
    ax.set_xlabel("Subscribed to Term Deposit?")
    ax.yaxis.set_major_formatter(mtick.FuncFormatter(lambda x, _: f"{int(x):,}"))
    fig.tight_layout()
    _save(fig, "01_target_distribution")
    return fig


# ── 2. Numeric feature distributions ─────────────────────────────────────────

def plot_numeric_distributions(df: pd.DataFrame) -> plt.Figure:
    """Histograms + KDE for each numeric feature, coloured by target."""
    num_cols = ["age", "balance", "campaign", "pdays", "previous", "was_contacted"]
    num_cols = [c for c in num_cols if c in df.columns]

    n = len(num_cols)
    ncols = 3
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(15, 4 * nrows))
    axes = axes.flatten()

    palette = {0: "#4878CF", 1: "#e05353"}
    labels = {0: "No", 1: "Yes"}

    for i, col in enumerate(num_cols):
        ax = axes[i]
        for val in [0, 1]:
            subset = df.loc[df["y"] == val, col]
            ax.hist(
                subset, bins=40, alpha=0.55,
                color=palette[val], label=labels[val], density=True,
            )
        ax.set_title(col, fontsize=11)
        ax.set_xlabel(col)
        ax.set_ylabel("Density")
        if i == 0:
            ax.legend(title="Subscribed")

    # Hide unused axes
    for j in range(n, len(axes)):
        axes[j].set_visible(False)

    fig.suptitle("Numeric Feature Distributions by Target", fontsize=14, fontweight="bold", y=1.01)
    fig.tight_layout()
    _save(fig, "02_numeric_distributions")
    return fig


# ── 3. Categorical feature distributions ──────────────────────────────────────

def plot_categorical_distributions(df: pd.DataFrame) -> plt.Figure:
    """Count plots for each categorical feature, split by target."""
    cat_cols = ["job", "marital", "education", "default", "housing",
                "loan", "contact", "month", "poutcome"]
    cat_cols = [c for c in cat_cols if c in df.columns]

    n = len(cat_cols)
    fig, axes = plt.subplots(n, 1, figsize=(14, 4 * n))

    palette = {0: "#4878CF", 1: "#e05353"}
    label_map = {0: "No", 1: "Yes"}

    for ax, col in zip(axes, cat_cols):
        # Compute order by total count
        order = df[col].value_counts().index.tolist()
        # Plot stacked percentage bars
        cross = df.groupby([col, "y"]).size().unstack(fill_value=0)
        cross_pct = cross.div(cross.sum(axis=1), axis=0) * 100
        cross_pct = cross_pct.reindex(order)
        cross_pct.rename(columns={0: "No", 1: "Yes"}).plot(
            kind="bar", stacked=True, ax=ax,
            color=["#4878CF", "#e05353"], edgecolor="white",
        )
        ax.set_title(f"{col} — Subscription Rate by Category", fontsize=11)
        ax.set_xlabel(col)
        ax.set_ylabel("% of customers")
        ax.set_xticklabels(ax.get_xticklabels(), rotation=30, ha="right", fontsize=9)
        ax.yaxis.set_major_formatter(mtick.PercentFormatter())
        ax.legend(title="Subscribed", loc="upper right", fontsize=8)

    fig.suptitle("Categorical Features vs Target (Subscription Rate)", fontsize=14, fontweight="bold")
    fig.tight_layout()
    _save(fig, "03_categorical_distributions")
    return fig


# ── 4. Correlation heatmap ────────────────────────────────────────────────────

def plot_correlation_heatmap(df: pd.DataFrame) -> plt.Figure:
    """Pearson correlation heatmap for numeric columns."""
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    corr = df[num_cols].corr()

    fig, ax = plt.subplots(figsize=(10, 8))
    mask = np.triu(np.ones_like(corr, dtype=bool))
    sns.heatmap(
        corr, mask=mask, annot=True, fmt=".2f",
        cmap="RdBu_r", center=0, linewidths=0.5,
        ax=ax, cbar_kws={"shrink": 0.8},
    )
    ax.set_title("Correlation Heatmap — Numeric Features", fontsize=13, fontweight="bold")
    fig.tight_layout()
    _save(fig, "04_correlation_heatmap")
    return fig


# ── 5. pdays analysis ─────────────────────────────────────────────────────────

def plot_pdays_analysis(df: pd.DataFrame) -> plt.Figure:
    """
    Two-panel figure:
      Left:  Pie chart — contacted vs never contacted proportions.
      Right: Subscription rate for contacted vs never contacted.
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Panel 1 — Contact proportion
    ax = axes[0]
    contacted = df["was_contacted"].value_counts().rename({0: "Never contacted\n(pdays=-1)", 1: "Previously contacted"})
    ax.pie(
        contacted.values, labels=contacted.index,
        autopct="%1.1f%%", startangle=90,
        colors=["#4878CF", "#e05353"], pctdistance=0.8,
    )
    ax.set_title("Prior Contact Distribution\n(was_contacted)", fontsize=11, fontweight="bold")

    # Panel 2 — Subscription rate
    ax2 = axes[1]
    sub_rate = (
        df.groupby("was_contacted")["y"]
        .mean()
        .rename({0: "Never\ncontacted", 1: "Previously\ncontacted"})
        * 100
    )
    bars = ax2.bar(sub_rate.index, sub_rate.values, color=["#4878CF", "#e05353"], edgecolor="white")
    for bar, val in zip(bars, sub_rate.values):
        ax2.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.5,
            f"{val:.1f}%", ha="center", va="bottom", fontsize=11, fontweight="bold",
        )
    ax2.set_title("Subscription Rate by Prior Contact Status", fontsize=11, fontweight="bold")
    ax2.set_ylabel("Subscription Rate (%)")
    ax2.set_ylim(0, sub_rate.max() * 1.25)
    ax2.yaxis.set_major_formatter(mtick.PercentFormatter())

    fig.suptitle("pdays Sentinel Analysis: Prior Contact vs Subscription", fontsize=13, fontweight="bold")
    fig.tight_layout()
    _save(fig, "05_pdays_analysis")
    return fig


# ── 6. Subscription rate by individual feature ────────────────────────────────

def plot_subscription_by_feature(df: pd.DataFrame, feature: str) -> plt.Figure:
    """Bar chart of subscription rate for each level of a given feature."""
    rate = df.groupby(feature)["y"].mean().sort_values(ascending=False) * 100

    fig, ax = plt.subplots(figsize=(max(6, len(rate) * 0.8), 4))
    bars = ax.bar(rate.index.astype(str), rate.values, color="#4878CF", edgecolor="white")
    for bar, val in zip(bars, rate.values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.3,
            f"{val:.1f}%", ha="center", va="bottom", fontsize=9,
        )
    ax.axhline(df["y"].mean() * 100, color="red", linestyle="--", linewidth=1, label="Overall rate")
    ax.set_title(f"Subscription Rate by '{feature}'", fontsize=12, fontweight="bold")
    ax.set_xlabel(feature)
    ax.set_ylabel("Subscription Rate (%)")
    ax.set_xticklabels(ax.get_xticklabels(), rotation=30, ha="right")
    ax.yaxis.set_major_formatter(mtick.PercentFormatter())
    ax.legend()
    fig.tight_layout()
    _save(fig, f"06_sub_rate_{feature}")
    return fig


# ── 7. Unknown audit visualisation ───────────────────────────────────────────

def plot_unknown_audit(df: pd.DataFrame) -> plt.Figure:
    """Bar chart showing the count of 'unknown' values per categorical column."""
    from src.data_loader import get_unknown_audit  # local import to avoid circular

    audit = get_unknown_audit(df)
    if audit.empty:
        fig, ax = plt.subplots(figsize=(5, 3))
        ax.text(0.5, 0.5, "No 'unknown' values found", ha="center", va="center", fontsize=12)
        ax.set_axis_off()
        _save(fig, "07_unknown_audit")
        return fig

    audit = audit.sort_values("unknown_count", ascending=True)
    fig, ax = plt.subplots(figsize=(8, max(3, len(audit) * 0.7)))
    bars = ax.barh(audit["column"], audit["unknown_count"], color="#e8a838", edgecolor="white")
    for bar, row in zip(bars, audit.itertuples()):
        ax.text(
            bar.get_width() + 100, bar.get_y() + bar.get_height() / 2,
            f"{row.unknown_count:,}  ({row.unknown_pct})",
            va="center", fontsize=9,
        )
    ax.set_title("'unknown' String Value Counts by Column", fontsize=12, fontweight="bold")
    ax.set_xlabel("Number of rows with 'unknown'")
    ax.xaxis.set_major_formatter(mtick.FuncFormatter(lambda x, _: f"{int(x):,}"))
    fig.tight_layout()
    _save(fig, "07_unknown_audit")
    return fig


# ── 8. Boxplots of numeric features by target ────────────────────────────────

def plot_numeric_boxplots(df: pd.DataFrame) -> plt.Figure:
    """Boxplots comparing numeric feature distributions for subscribers vs non-subscribers."""
    num_cols = ["age", "balance", "campaign", "previous"]
    num_cols = [c for c in num_cols if c in df.columns]

    n = len(num_cols)
    ncols = 2
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(12, 4 * nrows))
    axes = axes.flatten()

    for i, col in enumerate(num_cols):
        ax = axes[i]
        df_plot = df[["y", col]].copy()
        df_plot["Subscribed"] = df_plot["y"].map({0: "No", 1: "Yes"})
        sns.boxplot(data=df_plot, x="Subscribed", y=col, hue="Subscribed",
                    palette={"No": "#4878CF", "Yes": "#e05353"}, legend=False, ax=ax)
        ax.set_title(col, fontsize=11)

    for j in range(n, len(axes)):
        axes[j].set_visible(False)

    fig.suptitle("Numeric Feature Distributions by Subscription", fontsize=13, fontweight="bold")
    fig.tight_layout()
    _save(fig, "08_numeric_boxplots")
    return fig


if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from src.data_loader import load_data

    df = load_data()
    plot_target_distribution(df)
    plot_numeric_distributions(df)
    plot_categorical_distributions(df)
    plot_correlation_heatmap(df)
    plot_pdays_analysis(df)
    plot_unknown_audit(df)
    plot_numeric_boxplots(df)
    print("All EDA plots saved.")
    plt.close("all")
