"""Data loading and train/test splitting."""

from pathlib import Path
from typing import Tuple

import pandas as pd
from sklearn.model_selection import train_test_split

from phishdet.utils import get_logger

logger = get_logger(__name__)


def load_data(path: str) -> pd.DataFrame:
    """Load the URL dataset from a CSV file.

    The CSV must contain at least two columns:

    * ``url``   – raw URL string
    * ``label`` – 0 (legitimate) or 1 (phishing)

    Parameters
    ----------
    path:
        Path to the CSV file.

    Returns
    -------
    pd.DataFrame
        DataFrame with columns ``url`` and ``label``.

    Raises
    ------
    FileNotFoundError
        If *path* does not exist.
    ValueError
        If required columns are missing.
    """
    csv_path = Path(path)
    if not csv_path.exists():
        raise FileNotFoundError(f"Data file not found: {csv_path}")

    df = pd.read_csv(csv_path)

    missing = {"url", "label"} - set(df.columns)
    if missing:
        raise ValueError(f"CSV is missing required columns: {missing}")

    df = df[["url", "label"]].dropna()
    df["label"] = df["label"].astype(int)
    logger.info("Loaded %d rows from %s", len(df), csv_path)
    return df


def split_data(
    df: pd.DataFrame,
    test_size: float = 0.2,
    random_state: int = 42,
) -> Tuple[pd.Series, pd.Series, pd.Series, pd.Series]:
    """Split *df* into train and test subsets.

    Parameters
    ----------
    df:
        DataFrame with columns ``url`` and ``label``.
    test_size:
        Proportion of the dataset to include in the test split.
    random_state:
        Random seed for reproducibility.

    Returns
    -------
    tuple
        ``(X_train, X_test, y_train, y_test)`` where *X* is the URL series
        and *y* is the label series.
    """
    X = df["url"]
    y = df["label"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    logger.info(
        "Split: %d train / %d test samples", len(X_train), len(X_test)
    )
    return X_train, X_test, y_train, y_test
