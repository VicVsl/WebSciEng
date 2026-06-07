"""
Run a pretrained RoBERTa sentiment model over review texts and attach
the predicted label and confidence score to the DataFrame.

Model: cardiffnlp/twitter-roberta-base-sentiment-latest
Output labels: negative | neutral | positive  (matches our rating mapping)

Steps:
    1. load_model()          — download / cache the model, auto-detect GPU
    2. predict()             — run batched inference, add text_sentiment + score
    3. filter_low_confidence — drop rows the model is uncertain about
    4. save / load helpers   — persist results so the model doesn't re-run
"""

from pathlib import Path

import pandas as pd
import torch
from tqdm.auto import tqdm
from transformers import pipeline

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

MODEL_NAME = "cardiffnlp/twitter-roberta-base-sentiment-latest"

# Reviews whose top predicted sentiment has a score below this threshold
# are considered too uncertain and are dropped before mismatch detection.
# 0.60 means the model must assign at least 60% probability to its top label.
DEFAULT_CONFIDENCE_THRESHOLD = 0.60

# Number of reviews processed in one forward pass.
# Larger batches are faster but use more VRAM / RAM.
# 32 is safe for CPU; increase to 64–128 if you have a GPU.
DEFAULT_BATCH_SIZE = 32

SENTIMENT_PATH = Path("data/processed/reviews_with_sentiment.parquet")


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def load_model(batch_size: int = DEFAULT_BATCH_SIZE):
    """
    Load the RoBERTa sentiment pipeline from HuggingFace.

    The model is downloaded once and cached locally (~500 MB).
    Subsequent calls load from the local cache instantly.

    Parameters
    ----------
    batch_size : int
        How many reviews to process per forward pass. Increase if you have
        a GPU; keep at 32 or below for CPU to avoid memory issues.

    Returns
    -------
    transformers.Pipeline  (ready to call with a list of strings)
    """
    device = 0 if torch.cuda.is_available() else -1
    device_name = "GPU (cuda:0)" if device == 0 else "CPU"
    print(f"Loading sentiment model on {device_name}...")
    print(f"  Model : {MODEL_NAME}")
    print(f"  Batch size: {batch_size}")

    classifier = pipeline(  # type: ignore[call-overload]
        task="sentiment-analysis", # type: ignore
        model=MODEL_NAME,
        device=device,
        batch_size=batch_size,
        truncation=True,   # silently truncate reviews longer than 512 tokens
        max_length=512,
    )

    print("Model loaded.\n")
    return classifier


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------

def predict(
    df: pd.DataFrame,
    classifier,
    text_column: str = "text",
) -> pd.DataFrame:
    """
    Run the sentiment classifier over every review and attach results.

    Adds two new columns:
        text_sentiment  — predicted label: 'negative' | 'neutral' | 'positive'
        sentiment_score — confidence of the top prediction (0.0 – 1.0)

    Parameters
    ----------
    df : pd.DataFrame
        Preprocessed reviews (output of preprocessing.preprocess()).
    classifier : transformers.Pipeline
        Loaded model from load_model().
    text_column : str
        Column containing the review text to classify.

    Returns
    -------
    pd.DataFrame with text_sentiment and sentiment_score columns added.
    """
    texts = df[text_column].tolist()
    batch_size = getattr(classifier, "batch_size", DEFAULT_BATCH_SIZE)

    print(f"Running sentiment analysis on {len(texts):,} reviews...")

    results = []
    for start in tqdm(
        range(0, len(texts), batch_size),
        total=(len(texts) + batch_size - 1) // batch_size,
        desc="Sentiment inference",
        unit="batch",
    ):
        batch_texts = texts[start : start + batch_size]
        batch_results = classifier(batch_texts)
        for result in batch_results:
            results.append({
                "text_sentiment": result["label"].lower(),  # normalise to lowercase
                "sentiment_score": round(result["score"], 4),
            })

    sentiment_df = pd.DataFrame(results, index=df.index)
    df = pd.concat([df, sentiment_df], axis=1)

    print("\nPrediction complete.")
    print("Text sentiment distribution:")
    print(df["text_sentiment"].value_counts().to_string())
    print(f"\nMean confidence score: {df['sentiment_score'].mean():.3f}")

    return df


# ---------------------------------------------------------------------------
# Confidence filtering
# ---------------------------------------------------------------------------

def filter_low_confidence(
    df: pd.DataFrame,
    threshold: float = DEFAULT_CONFIDENCE_THRESHOLD,
) -> pd.DataFrame:
    """
    Remove reviews where the model's confidence is below `threshold`.

    Per the paper's methodology: reviews where sentiment cannot be captured
    at a high enough degree of confidence are excluded from the analysis.
    This avoids treating genuinely ambiguous cases as clean mismatches.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with a 'sentiment_score' column (output of predict()).
    threshold : float
        Minimum confidence to keep a row. Default is 0.60.

    Returns
    -------
    Filtered pd.DataFrame.
    """
    before = len(df)
    df = df[df["sentiment_score"] >= threshold].reset_index(drop=True)
    removed = before - len(df)
    pct = removed / before * 100 if before > 0 else 0
    print(
        f"[filter_low_confidence (threshold={threshold})] "
        f"{before:,} → {len(df):,}  (removed {removed:,} = {pct:.1f}%)"
    )
    return df


# ---------------------------------------------------------------------------
# Save / load helpers
# ---------------------------------------------------------------------------

def save_with_sentiment(
    df: pd.DataFrame,
    path: Path | str = SENTIMENT_PATH,
) -> None:
    """Save the DataFrame (with sentiment columns) to parquet."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False)
    print(f"Saved to {path}  ({len(df):,} rows)")


def load_with_sentiment(
    path: Path | str = SENTIMENT_PATH,
) -> pd.DataFrame:
    """Load a previously saved sentiment-annotated parquet file."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"No sentiment data found at {path}. "
            "Run predict() + save_with_sentiment() first."
        )
    df = pd.read_parquet(path)
    print(f"Loaded {len(df):,} reviews with sentiment from {path}")
    return df


def sentiment_data_exists(path: Path | str = SENTIMENT_PATH) -> bool:
    """Check whether saved sentiment results already exist."""
    return Path(path).exists()