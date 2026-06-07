"""
Compare text_sentiment against rating_sentiment and classify each review.

Mismatch types (from the paper):
    match          — text and rating sentiment are the same
    soft_mismatch  — sentiments differ by one level  (e.g. positive vs neutral)
    strong_mismatch— sentiments differ by two levels (e.g. positive vs negative)

Sentiment scale used for distance calculation:
    negative = 0  |  neutral = 1  |  positive = 2

Added columns:
    sentiment_gap       int   (0, 1, or 2) — distance between the two sentiments
    mismatch_type       str   'match' | 'soft_mismatch' | 'strong_mismatch'
    is_mismatch         bool  True for soft and strong mismatches
    mismatch_direction  str   'text_more_positive' | 'text_more_negative' | 'none'
"""

from pathlib import Path

import pandas as pd

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Numeric encoding of sentiment levels — used to compute the gap between
# text_sentiment and rating_sentiment.
SENTIMENT_ORDER: dict[str, int] = {
    "negative": 0,
    "neutral":  1,
    "positive": 2,
}

# Reverse mapping for readability in outputs
ORDER_TO_SENTIMENT: dict[int, str] = {v: k for k, v in SENTIMENT_ORDER.items()}

MISMATCH_PATH = Path("data/processed/reviews_with_mismatch.parquet")


# ---------------------------------------------------------------------------
# Classification
# ---------------------------------------------------------------------------

def classify(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add mismatch classification columns to a sentiment-annotated DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain 'text_sentiment' and 'rating_sentiment' columns
        (output of sentiment.predict() and preprocessing.add_rating_label()).

    Returns
    -------
    pd.DataFrame with four new columns:
        sentiment_gap, mismatch_type, is_mismatch, mismatch_direction
    """
    df = df.copy()

    # Encode both sentiments as integers so we can compute the gap
    text_score   = df["text_sentiment"].map(SENTIMENT_ORDER)
    rating_score = df["rating_sentiment"].map(SENTIMENT_ORDER)

    gap = (text_score - rating_score)

    df["sentiment_gap"] = gap.abs()

    df["mismatch_type"] = df["sentiment_gap"].map({
        0: "match",
        1: "soft_mismatch",
        2: "strong_mismatch",
    })

    df["is_mismatch"] = df["sentiment_gap"] > 0

    # Direction: did the reviewer write more positively or more negatively
    # than their star rating implied?  Useful for qualitative interpretation.
    df["mismatch_direction"] = pd.Categorical(
        gap.map(lambda g: (
            "text_more_positive" if g > 0
            else "text_more_negative" if g < 0
            else "none"
        )),
        categories=["text_more_positive", "none", "text_more_negative"],
    )

    print("Mismatch classification complete.")
    print(f"\nOverall mismatch rate: {df['is_mismatch'].mean():.1%}")
    print("\nMismatch type counts:")
    print(df["mismatch_type"].value_counts().to_string())

    return df


# ---------------------------------------------------------------------------
# Summary helpers
# ---------------------------------------------------------------------------

def summary(df: pd.DataFrame) -> pd.DataFrame:
    """
    Return a tidy summary table of mismatch rates broken down by category
    and mismatch type.

    Columns in the returned table:
        category | mismatch_type | count | pct_of_category
    """
    total_by_cat = df.groupby("category").size().rename("total")

    counts = (
        df.groupby(["category", "mismatch_type"])
          .size()
          .rename("count")
          .reset_index()
    )

    counts = counts.merge(total_by_cat, on="category")
    counts["pct_of_category"] = (counts["count"] / counts["total"] * 100).round(2)
    counts = counts.drop(columns="total")

    # Sort by category then a logical mismatch order
    type_order = ["match", "soft_mismatch", "strong_mismatch"]
    counts["mismatch_type"] = pd.Categorical(
        counts["mismatch_type"], categories=type_order, ordered=True
    )
    counts = counts.sort_values(["category", "mismatch_type"]).reset_index(drop=True)

    return counts


def summary_by_rating(df: pd.DataFrame) -> pd.DataFrame:
    """
    Return a summary table of mismatch rates broken down by star rating,
    useful for checking whether certain ratings mismatch more often.

    Columns: rating | mismatch_type | count | pct_of_rating
    """
    total_by_rating = df.groupby("rating_int").size().rename("total")

    counts = (
        df.groupby(["rating_int", "mismatch_type"])
          .size()
          .rename("count")
          .reset_index()
    )

    counts = counts.merge(total_by_rating, on="rating_int")
    counts["pct_of_rating"] = (counts["count"] / counts["total"] * 100).round(2)
    counts = counts.drop(columns="total")

    type_order = ["match", "soft_mismatch", "strong_mismatch"]
    counts["mismatch_type"] = pd.Categorical(
        counts["mismatch_type"], categories=type_order, ordered=True
    )
    counts = counts.sort_values(["rating_int", "mismatch_type"]).reset_index(drop=True)

    return counts


def sample_mismatches(
    df: pd.DataFrame,
    mismatch_type: str = "strong_mismatch",
    n: int = 10,
    category: str | None = None,
    seed: int = 42,
) -> pd.DataFrame:
    """
    Return a random sample of mismatch cases for qualitative inspection.

    Parameters
    ----------
    df : pd.DataFrame
        Classified DataFrame (output of classify()).
    mismatch_type : str
        One of 'soft_mismatch' or 'strong_mismatch'.
    n : int
        Number of examples to return.
    category : str | None
        If provided, restrict to a single category.
    seed : int
        Random seed for reproducibility.

    Returns
    -------
    pd.DataFrame with columns: category, rating, rating_sentiment,
                                text_sentiment, sentiment_score,
                                mismatch_direction, text
    """
    mask = df["mismatch_type"] == mismatch_type
    if category:
        mask &= df["category"] == category

    sample = (
        df[mask]
        .sample(n=min(n, mask.sum()), random_state=seed)
        [["category", "rating", "rating_sentiment",
          "text_sentiment", "sentiment_score",
          "mismatch_direction", "text"]]
        .reset_index(drop=True)
    )

    return sample


# ---------------------------------------------------------------------------
# Save / load helpers
# ---------------------------------------------------------------------------

def save_classified(
    df: pd.DataFrame,
    path: Path | str = MISMATCH_PATH,
) -> None:
    """Save the fully classified DataFrame to parquet."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False)
    print(f"Saved classified data to {path}  ({len(df):,} rows)")


def load_classified(path: Path | str = MISMATCH_PATH) -> pd.DataFrame:
    """Load a previously saved classified parquet file."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"No classified data found at {path}. "
            "Run classify() + save_classified() first."
        )
    df = pd.read_parquet(path)
    print(f"Loaded {len(df):,} classified reviews from {path}")
    return df


def classified_data_exists(path: Path | str = MISMATCH_PATH) -> bool:
    """Check whether saved classified results already exist."""
    return Path(path).exists()