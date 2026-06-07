"""
Clean and prepare raw Amazon reviews for sentiment analysis.

Steps (applied in order by preprocess()):
    1. drop_missing       — remove rows without both a rating and review text
    2. drop_duplicates    — keep only a user's first review per category
    3. clean_text         — strip whitespace, normalise newlines
    4. drop_short         — remove reviews too short for reliable sentiment
    5. filter_english     — remove non-English reviews
    6. add_rating_label   — map star rating → negative / neutral / positive
"""

from pathlib import Path

import pandas as pd
from langdetect import detect, LangDetectException
from langdetect import DetectorFactory
from tqdm import tqdm

# Make langdetect deterministic across runs
DetectorFactory.seed = 42

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Minimum number of words a review must have for sentiment to be meaningful.
# Reviews shorter than this are often uninformative ("Great!", "Terrible.")
# and return low-confidence predictions from the model.
MIN_WORDS = 10

# Rating → sentiment label mapping (as defined in the paper)
RATING_TO_SENTIMENT: dict[int, str] = {
    1: "negative",
    2: "negative",
    3: "neutral",
    4: "positive",
    5: "positive",
}

PROCESSED_PATH = Path("data/processed/reviews_processed.parquet")


# ---------------------------------------------------------------------------
# Individual preprocessing steps
# ---------------------------------------------------------------------------

def drop_missing(df: pd.DataFrame) -> pd.DataFrame:
    """
    Remove rows that are missing a star rating or review text.
    Both are required — without either, mismatch detection is impossible.
    """
    before = len(df)
    df = df.dropna(subset=["rating", "text"])
    df = df[df["text"].str.strip().ne("")]          # drop blank strings too
    df = df[df["rating"].between(1, 5)]             # guard against bad values
    _report("drop_missing", before, len(df))
    return df.reset_index(drop=True)


def drop_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    """
    For each (user_id, category) pair keep only the earliest review.
    This follows the paper's rule of one review per user to avoid a single
    prolific reviewer skewing the mismatch statistics.
    """
    before = len(df)

    # Sort so the earliest review comes first, then deduplicate
    df = (
        df.sort_values("timestamp", ascending=True)
          .drop_duplicates(subset=["user_id", "category"], keep="first")
    )

    _report("drop_duplicates", before, len(df))
    return df.reset_index(drop=True)


def clean_text(df: pd.DataFrame) -> pd.DataFrame:
    """
    Normalise review text:
      - strip leading / trailing whitespace
      - collapse multiple newlines / tabs into a single space
    The text is not lowercased here because the sentiment model is
    case-sensitive and expects natural-casing input.
    """
    df = df.copy()
    df["text"] = (
        df["text"]
        .str.strip()
        .str.replace(r"\s+", " ", regex=True)
    )
    return df


def drop_short(df: pd.DataFrame, min_words: int = MIN_WORDS) -> pd.DataFrame:
    """
    Remove reviews whose text contains fewer than `min_words` words.
    Very short reviews (e.g. 'Great product!') rarely carry enough signal
    for a transformer model to produce a confident sentiment prediction.
    """
    before = len(df)
    word_counts = df["text"].str.split().str.len()
    df = df[word_counts >= min_words]
    _report(f"drop_short (< {min_words} words)", before, len(df))
    return df.reset_index(drop=True)


def filter_english(df: pd.DataFrame) -> pd.DataFrame:
    """
    Retain only English-language reviews.
    Uses langdetect on the first 200 characters of each review (faster than
    running on the full text, and sufficient for language identification).
    Rows where detection fails are dropped to be safe.
    """
    before = len(df)

    def is_english(text: str) -> bool:
        try:
            return detect(text[:200]) == "en"
        except LangDetectException:
            return False

    tqdm.pandas(desc="Detecting language")
    mask = df["text"].progress_apply(is_english)
    df = df[mask]

    _report("filter_english", before, len(df))
    return df.reset_index(drop=True)


def add_rating_label(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add a 'rating_sentiment' column by mapping the integer star rating to
    one of: 'negative' | 'neutral' | 'positive'.

    Mapping (defined in the paper's methodology):
        1-2 stars  →  negative
        3 stars    →  neutral
        4-5 stars  →  positive
    """
    df = df.copy()
    df["rating_int"] = df["rating"].round().astype(int)
    df["rating_sentiment"] = df["rating_int"].map(RATING_TO_SENTIMENT)
    return df


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def preprocess(df: pd.DataFrame, min_words: int = MIN_WORDS) -> pd.DataFrame:
    """
    Run the full preprocessing pipeline on a raw reviews DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
        Raw output from data_loader.load_categories().
    min_words : int
        Minimum word count for a review to be kept (default: 10).

    Returns
    -------
    pd.DataFrame
        Cleaned DataFrame with an added 'rating_sentiment' column,
        ready to be passed to the sentiment analysis step.
    """
    print("Starting preprocessing pipeline...")
    print(f"  Input: {len(df):,} reviews\n")

    df = drop_missing(df)
    df = drop_duplicates(df)
    df = clean_text(df)
    df = drop_short(df, min_words=min_words)
    df = filter_english(df)
    df = add_rating_label(df)

    print(f"\nPreprocessing complete.")
    print(f"  Output: {len(df):,} reviews")
    print(f"\nRating sentiment distribution:")
    print(df["rating_sentiment"].value_counts().to_string())
    print(f"\nBy category:")
    print(df.groupby("category")["rating_sentiment"].value_counts().to_string())

    return df


# ---------------------------------------------------------------------------
# Save / load helpers
# ---------------------------------------------------------------------------

def save_processed(df: pd.DataFrame, path: Path | str = PROCESSED_PATH) -> None:
    """Save the cleaned DataFrame to parquet."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False)
    print(f"Processed data saved to {path}  ({len(df):,} rows)")


def load_processed(path: Path | str = PROCESSED_PATH) -> pd.DataFrame:
    """Load a previously saved processed parquet file."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"No processed data found at {path}. "
            "Run preprocess() + save_processed() first."
        )
    df = pd.read_parquet(path)
    print(f"Loaded {len(df):,} processed reviews from {path}")
    return df


def processed_data_exists(path: Path | str = PROCESSED_PATH) -> bool:
    """Check whether a saved processed file already exists."""
    return Path(path).exists()


# ---------------------------------------------------------------------------
# Internal helper
# ---------------------------------------------------------------------------

def _report(step: str, before: int, after: int) -> None:
    """Print a one-line summary of how many rows a step removed."""
    removed = before - after
    pct = (removed / before * 100) if before > 0 else 0
    print(f"  [{step}] {before:,} → {after:,}  (removed {removed:,} = {pct:.1f}%)")