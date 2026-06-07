"""
Stream a balanced sample of Amazon Reviews 2023 from HuggingFace.
Dataset: McAuley-Lab/Amazon-Reviews-2023

No manual download needed — data is streamed on demand and then saved
locally as parquet so subsequent runs skip the download entirely.
"""

import itertools
from pathlib import Path

import pandas as pd
from datasets import load_dataset

# ---------------------------------------------------------------------------
# Category registry
# Maps friendly names -> direct JSONL paths on the HuggingFace Hub.
#
# We load the raw JSONL files directly using the built-in "json" loader
# instead of the dataset's custom loading script, because datasets>=3.0
# no longer supports custom scripts. The hf:// prefix tells the datasets
# library to resolve the path from the Hub without any script.
# ---------------------------------------------------------------------------
_HF_BASE = "hf://datasets/McAuley-Lab/Amazon-Reviews-2023/raw/review_categories"

CATEGORY_CONFIGS: dict[str, str] = {
    "Electronics":  f"{_HF_BASE}/Electronics.jsonl",
    "Clothing":     f"{_HF_BASE}/Clothing_Shoes_and_Jewelry.jsonl",
    "Books":                f"{_HF_BASE}/Books.jsonl",
    "Home_and_Kitchen":     f"{_HF_BASE}/Home_and_Kitchen.jsonl",
    "Sports_and_Outdoors":  f"{_HF_BASE}/Sports_and_Outdoors.jsonl",
    "Automotive":           f"{_HF_BASE}/Automotive.jsonl",
}

# Columns to keep from each raw review record
KEEP_COLUMNS = [
    "rating",            # float: 1.0 – 5.0
    "text",              # str:   body of the review
    "title",             # str:   short headline the reviewer wrote
    "user_id",           # str:   anonymised reviewer ID (used for deduplication)
    "asin",              # str:   product ID
    "timestamp",         # int:   Unix timestamp in milliseconds
    "helpful_vote",      # int:   number of "helpful" votes
    "verified_purchase", # bool:  whether Amazon verified the purchase
]

RAW_PATH = Path("data/raw/reviews_raw.parquet")


# ---------------------------------------------------------------------------
# Core loading functions
# ---------------------------------------------------------------------------

def load_category(
    category: str,
    n_samples: int = 5000,
    seed: int = 42,
    shuffle_buffer: int = 50_000,
) -> pd.DataFrame:
    """
    Stream a random sample of reviews for a single category.

    Parameters
    ----------
    category : str
        Friendly name (must be a key in CATEGORY_CONFIGS).
    n_samples : int
        How many reviews to sample. Keep at <=5000 per category when running
        on CPU — the sentiment model in the next step is the bottleneck.
    seed : int
        Random seed for the streaming shuffle (ensures reproducibility).
    shuffle_buffer : int
        How many rows the streamer buffers before drawing randomly.
        Larger = more representative sample, but uses more RAM during loading.

    Returns
    -------
    pd.DataFrame
        Columns: rating, text, title, user_id, asin, timestamp,
                 helpful_vote, verified_purchase, category
    """
    if category not in CATEGORY_CONFIGS:
        available = list(CATEGORY_CONFIGS.keys())
        raise ValueError(
            f"Unknown category '{category}'. Available: {available}"
        )

    jsonl_path = CATEGORY_CONFIGS[category]
    print(f"[{category}] Connecting to HuggingFace...")

    # Load directly from the JSONL file — bypasses the (now unsupported)
    # custom loading script entirely.
    ds = load_dataset(
        "json",
        data_files={"full": jsonl_path},
        split="full",
        streaming=True,
    )

    # Shuffle before sampling so we get reviews spread across time/products,
    # not just the first n_samples rows of the file.
    ds = ds.shuffle(seed=seed, buffer_size=shuffle_buffer)

    print(f"[{category}] Sampling {n_samples} reviews...")
    records = [
        {col: row.get(col) for col in KEEP_COLUMNS}
        for row in itertools.islice(ds, n_samples)
    ]

    df = pd.DataFrame(records)
    df["category"] = category
    df["rating"] = pd.to_numeric(df["rating"], errors="coerce")

    print(f"[{category}] Done — {len(df):,} reviews loaded.")
    return df


def load_categories(
    categories: list[str] | None = None,
    n_samples_per_category: int = 5000,
    seed: int = 42,
) -> pd.DataFrame:
    """
    Load a balanced sample across multiple categories and combine into
    a single DataFrame.

    Parameters
    ----------
    categories : list[str] | None
        Category names to load. Defaults to all entries in CATEGORY_CONFIGS.
    n_samples_per_category : int
        Rows to sample per category. The same seed is used for all categories
        so results are reproducible.
    seed : int
        Passed to load_category for every category.

    Returns
    -------
    pd.DataFrame with a 'category' column identifying each review's source.
    """
    if categories is None:
        categories = list(CATEGORY_CONFIGS.keys())

    frames = [
        load_category(cat, n_samples=n_samples_per_category, seed=seed)
        for cat in categories
    ]

    combined = pd.concat(frames, ignore_index=True)

    print(f"\n{'='*40}")
    print(f"Total reviews loaded: {len(combined):,}")
    print(combined["category"].value_counts().to_string())
    print(f"{'='*40}\n")

    return combined


# ---------------------------------------------------------------------------
# Save / load helpers  (avoids re-streaming on every notebook run)
# ---------------------------------------------------------------------------

def save_raw(df: pd.DataFrame, path: Path | str = RAW_PATH) -> None:
    """
    Persist the raw combined DataFrame as parquet.
    Call this once after load_categories() so subsequent runs can use load_raw().
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False)
    print(f"Raw data saved to {path}  ({len(df):,} rows)")


def load_raw(path: Path | str = RAW_PATH) -> pd.DataFrame:
    """
    Load the previously saved raw parquet file.
    Much faster than re-streaming from HuggingFace.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"No raw data found at {path}. "
            "Run load_categories() + save_raw() first."
        )
    df = pd.read_parquet(path)
    print(f"Loaded {len(df):,} reviews from {path}")
    return df


def raw_data_exists(path: Path | str = RAW_PATH) -> bool:
    """Check whether a saved raw file already exists."""
    return Path(path).exists()