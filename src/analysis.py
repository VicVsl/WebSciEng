"""
src/analysis.py

Generate plots and tables for the mismatch analysis paper.

Quantitative:
    plot_mismatch_by_category  — mismatch rates per category (grouped bar)
    plot_mismatch_by_rating    — mismatch rates per star rating (stacked bar)
    plot_mismatch_direction    — text-more-positive vs text-more-negative by category

Tables:
    save_summary_tables        — CSV + LaTeX for the category and rating summaries

Qualitative:
    qualitative_table          — formatted DataFrame of mismatch examples for inspection

All plots are saved to results/figures/ and all tables to results/tables/.

Usage
-----
from src.analysis import (
    plot_mismatch_by_category,
    plot_mismatch_by_rating,
    plot_mismatch_direction,
    save_summary_tables,
    qualitative_table,
)
"""

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.figure import Figure
import matplotlib.ticker as mtick
import pandas as pd
import seaborn as sns

from scipy.stats import chi2_contingency, mannwhitneyu
import numpy as np

# ---------------------------------------------------------------------------
# Style
# ---------------------------------------------------------------------------

# Consistent palette across all figures
PALETTE = {
    "match":           "#4C9A6F",   # green
    "soft_mismatch":   "#E8A838",   # amber
    "strong_mismatch": "#C0392B",   # red
}

DIRECTION_PALETTE = {
    "text_more_positive": "#4C9A6F",
    "none":               "#AAAAAA",
    "text_more_negative": "#C0392B",
}

TYPE_ORDER    = ["match", "soft_mismatch", "strong_mismatch"]
TYPE_LABELS   = ["Match", "Soft mismatch", "Strong mismatch"]
DIR_ORDER     = ["text_more_positive", "none", "text_more_negative"]
DIR_LABELS    = ["Text more positive", "No mismatch", "Text more negative"]

FIGURES_DIR = Path("results/figures")
TABLES_DIR  = Path("results/tables")


def _setup_style() -> None:
    """Apply a clean, publication-friendly matplotlib style."""
    sns.set_theme(style="whitegrid", font_scale=1.1)
    plt.rcParams.update({
        "figure.dpi":      150,
        "axes.spines.top": False,
        "axes.spines.right": False,
    })


def _save(fig: Figure, filename: str) -> None:
    """Save a figure to results/figures/ as both PNG and PDF."""
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        out = FIGURES_DIR / f"{filename}.{ext}"
        fig.savefig(out, bbox_inches="tight")
    print(f"Saved figure → {FIGURES_DIR / filename}.png / .pdf")


# ---------------------------------------------------------------------------
# Statistical analysis helpers
# ---------------------------------------------------------------------------

def sentiment_confusion_matrix(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create a row-normalized confusion matrix comparing rating sentiment
    against text sentiment.
    """
    cm = pd.crosstab(
        df["rating_sentiment"],
        df["text_sentiment"],
        normalize="index",
    ) * 100

    return cm.round(2)


def cramers_v(contingency: pd.DataFrame) -> float:
    """
    Compute Cramér's V effect size from a contingency table.
    """
    chi2 = chi2_contingency(contingency)[0]
    n = contingency.to_numpy().sum()
    r, k = contingency.shape

    return np.sqrt(chi2 / (n * (min(r, k) - 1)))


def chi_square_category_test(df: pd.DataFrame) -> dict:
    """
    Test whether mismatch rates differ across categories.
    """
    table = pd.crosstab(df["category"], df["is_mismatch"])

    chi2, p, dof, _ = chi2_contingency(table)

    return {
        "chi2": chi2,
        "p": p,
        "dof": dof,
        "cramers_v": cramers_v(table),
    }



def chi_square_rating_test(df: pd.DataFrame) -> dict:
    """
    Test whether mismatch rates differ across rating levels.
    """
    table = pd.crosstab(df["rating_int"], df["is_mismatch"])

    chi2, p, dof, _ = chi2_contingency(table)

    return {
        "chi2": chi2,
        "p": p,
        "dof": dof,
        "cramers_v": cramers_v(table),
    }



def confidence_comparison(df: pd.DataFrame) -> dict:
    """
    Compare sentiment-model confidence between matched and mismatched reviews.
    """
    matches = df[~df["is_mismatch"]]["sentiment_score"]
    mismatches = df[df["is_mismatch"]]["sentiment_score"]

    statistic, p = mannwhitneyu(
        matches,
        mismatches,
        alternative="two-sided",
    )

    return {
        "match_mean": matches.mean(),
        "mismatch_mean": mismatches.mean(),
        "u_statistic": statistic,
        "p": p,
    }



def mismatch_direction_table(df: pd.DataFrame) -> pd.DataFrame:
    """
    Percentage breakdown of mismatch direction by category.
    """
    mismatches = df[df["is_mismatch"]]

    table = (
        pd.crosstab(
            mismatches["category"],
            mismatches["mismatch_direction"],
            normalize="index",
        ) * 100
    )

    return table.round(2)



def review_length_analysis(df: pd.DataFrame) -> dict:
    """
    Compare review lengths between matches and mismatches.
    """
    data = df.copy()
    data["review_length"] = data["text"].str.split().str.len()

    matches = data[~data["is_mismatch"]]["review_length"]
    mismatches = data[data["is_mismatch"]]["review_length"]

    statistic, p = mannwhitneyu(
        matches,
        mismatches,
        alternative="two-sided",
    )

    return {
        "match_mean_length": matches.mean(),
        "mismatch_mean_length": mismatches.mean(),
        "u_statistic": statistic,
        "p": p,
    }


# ---------------------------------------------------------------------------
# Plot 1: Mismatch rate by category
# ---------------------------------------------------------------------------

def plot_mismatch_by_category(df: pd.DataFrame, save: bool = True) -> Figure:
    """
    Grouped bar chart showing the percentage of matches, soft mismatches,
    and strong mismatches for each product category.

    This is the core quantitative comparison across categories.
    """
    _setup_style()

    # Compute percentages
    counts = (
        df.groupby(["category", "mismatch_type"])
          .size()
          .reset_index(name="count")
    )
    totals = df.groupby("category").size().reset_index(name="total")
    counts = counts.merge(totals, on="category")
    counts["pct"] = counts["count"] / counts["total"] * 100
    counts["mismatch_type"] = pd.Categorical(
        counts["mismatch_type"], categories=TYPE_ORDER, ordered=True
    )
    counts = counts.sort_values("mismatch_type")

    fig, ax = plt.subplots(figsize=(8, 5))

    categories = df["category"].unique()
    x = range(len(categories))
    bar_width = 0.25
    offsets = [-bar_width, 0, bar_width]

    for offset, mtype, mlabel in zip(offsets, TYPE_ORDER, TYPE_LABELS):
        type_counts = counts[counts["mismatch_type"] == mtype]
        values = []
        for cat in categories:
            match = type_counts[type_counts["category"] == cat]
            values.append(match["pct"].iloc[0] if not match.empty else 0)
        ax.bar(
            [xi + offset for xi in x],
            values,
            width=bar_width,
            label=mlabel,
            color=PALETTE[mtype],
            edgecolor="white",
            linewidth=0.5,
        )

    ax.set_xticks(list(x))
    ax.set_xticklabels(categories, fontsize=11)
    ax.yaxis.set_major_formatter(mtick.PercentFormatter())
    ax.set_ylabel("Percentage of reviews")
    ax.set_title("Mismatch rate by product category", fontsize=13, pad=12)
    ax.legend(frameon=False)

    fig.tight_layout()
    if save:
        _save(fig, "mismatch_by_category")
    return fig


# ---------------------------------------------------------------------------
# Plot 2: Mismatch rate by star rating
# ---------------------------------------------------------------------------

def plot_mismatch_by_rating(df: pd.DataFrame, save: bool = True) -> Figure:
    """
    Stacked bar chart showing mismatch composition for each star rating (1–5).

    Directly tests the paper's expectation that 3-star reviews are the
    most ambiguous and therefore most prone to mismatches.
    """
    _setup_style()

    counts = (
        df.groupby(["rating_int", "mismatch_type"])
          .size()
          .reset_index(name="count")
    )
    totals = df.groupby("rating_int").size().reset_index(name="total")
    counts = counts.merge(totals, on="rating_int")
    counts["pct"] = counts["count"] / counts["total"] * 100

    pivot = (
        counts.pivot(index="rating_int", columns="mismatch_type", values="pct")
              .reindex(columns=TYPE_ORDER)
              .fillna(0)
    )

    fig, ax = plt.subplots(figsize=(8, 5))

    bottom = [0.0] * len(pivot)
    for mtype, mlabel in zip(TYPE_ORDER, TYPE_LABELS):
        values = pivot[mtype].tolist()
        ax.bar(
            pivot.index,
            values,
            bottom=bottom,
            label=mlabel,
            color=PALETTE[mtype],
            edgecolor="white",
            linewidth=0.5,
            width=0.6,
        )
        bottom = [b + v for b, v in zip(bottom, values)]

    ax.set_xlabel("Star rating")
    ax.set_ylabel("Percentage of reviews")
    ax.set_title("Mismatch composition by star rating", fontsize=13, pad=12)
    ax.yaxis.set_major_formatter(mtick.PercentFormatter())
    ax.set_xticks(pivot.index)
    ax.set_xticklabels([f"{r}★" for r in pivot.index])
    ax.legend(frameon=False, loc="upper right")

    fig.tight_layout()
    if save:
        _save(fig, "mismatch_by_rating")
    return fig


# ---------------------------------------------------------------------------
# Plot 3: Mismatch direction by category
# ---------------------------------------------------------------------------

def plot_mismatch_direction(df: pd.DataFrame, save: bool = True) -> Figure:
    """
    Stacked bar chart showing — for mismatched reviews only — whether the
    written text was more positive or more negative than the star rating,
    broken down by category.

    Helps characterise the *nature* of mismatches qualitatively.
    """
    _setup_style()

    mismatches = df[df["is_mismatch"]].copy()

    counts = (
        mismatches.groupby(["category", "mismatch_direction"])
                  .size()
                  .reset_index(name="count")
    )
    totals = mismatches.groupby("category").size().reset_index(name="total")
    counts = counts.merge(totals, on="category")
    counts["pct"] = counts["count"] / counts["total"] * 100

    pivot = (
        counts.pivot(index="category", columns="mismatch_direction", values="pct")
              .reindex(columns=["text_more_positive", "text_more_negative"])
              .fillna(0)
    )

    fig, ax = plt.subplots(figsize=(8, 5))

    bottom = [0.0] * len(pivot)
    for direction, label in zip(
        ["text_more_positive", "text_more_negative"],
        ["Text more positive than rating", "Text more negative than rating"],
    ):
        values = pivot[direction].tolist()
        ax.bar(
            pivot.index,
            values,
            bottom=bottom,
            label=label,
            color=DIRECTION_PALETTE[direction],
            edgecolor="white",
            linewidth=0.5,
            width=0.5,
        )
        bottom = [b + v for b, v in zip(bottom, values)]

    ax.set_ylabel("Percentage of mismatched reviews")
    ax.set_title("Mismatch direction by category\n(mismatched reviews only)",
                 fontsize=13, pad=12)
    ax.yaxis.set_major_formatter(mtick.PercentFormatter())
    ax.legend(frameon=False)

    fig.tight_layout()
    if save:
        _save(fig, "mismatch_direction_by_category")
    return fig


# ---------------------------------------------------------------------------
# Tables
# ---------------------------------------------------------------------------

def save_summary_tables(
    df: pd.DataFrame,
    from_mismatch_module: bool = True,
) -> None:
    """
    Save the category-level and rating-level summary tables as CSV and LaTeX.
    Both are written to results/tables/.

    Parameters
    ----------
    df : pd.DataFrame
        Fully classified DataFrame (output of mismatch.classify()).
    from_mismatch_module : bool
        If True (default), imports summary helpers from mismatch.py.
        Set to False only if you want to pass pre-built tables manually.
    """
    from src.mismatch import summary, summary_by_rating

    TABLES_DIR.mkdir(parents=True, exist_ok=True)

    # --- Category summary ---
    cat_table = summary(df)
    cat_table.to_csv(TABLES_DIR / "summary_by_category.csv", index=False)
    cat_table.to_latex(
        TABLES_DIR / "summary_by_category.tex",
        index=False,
        float_format="%.2f",
        caption="Mismatch rates by product category.",
        label="tab:mismatch_by_category",
    )
    print(f"Saved summary_by_category  → {TABLES_DIR}")

    # --- Rating summary ---
    rating_table = summary_by_rating(df)
    rating_table.to_csv(TABLES_DIR / "summary_by_rating.csv", index=False)
    rating_table.to_latex(
        TABLES_DIR / "summary_by_rating.tex",
        index=False,
        float_format="%.2f",
        caption="Mismatch rates by star rating.",
        label="tab:mismatch_by_rating",
    )
    print(f"Saved summary_by_rating    → {TABLES_DIR}")


# ---------------------------------------------------------------------------
# Qualitative helper
# ---------------------------------------------------------------------------

def qualitative_table(
    df: pd.DataFrame,
    mismatch_type: str = "strong_mismatch",
    n: int = 10,
    category: str | None = None,
    seed: int = 42,
    truncate_text: int = 200,
) -> pd.DataFrame:
    """
    Return a neatly formatted sample of mismatch cases for qualitative
    inspection in the notebook or for inclusion in the paper's appendix.

    Parameters
    ----------
    df : pd.DataFrame
        Classified DataFrame.
    mismatch_type : str
        'soft_mismatch' or 'strong_mismatch'.
    n : int
        Number of examples to show.
    category : str | None
        Filter to a single category if provided.
    seed : int
        Reproducibility seed.
    truncate_text : int
        Max characters of review text to display (keeps tables readable).
    """
    from src.mismatch import sample_mismatches

    sample = sample_mismatches(df, mismatch_type=mismatch_type,
                               n=n, category=category, seed=seed)

    sample = sample.copy()
    sample["text"] = sample["text"].str[:truncate_text] + "…"
    sample = sample.rename(columns={
        "rating":            "Stars",
        "rating_sentiment":  "Rating sentiment",
        "text_sentiment":    "Text sentiment",
        "sentiment_score":   "Confidence",
        "mismatch_direction":"Direction",
        "text":              "Review excerpt",
    })

    return sample