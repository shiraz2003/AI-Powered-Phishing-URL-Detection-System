"""Random-Forest based phishing URL classifier."""

from pathlib import Path
from typing import List, Union

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

from phishdet.features import build_feature_matrix
from phishdet.utils import get_logger

logger = get_logger(__name__)

_DEFAULT_PARAMS = {
    "n_estimators": 100,
    "max_depth": None,
    "random_state": 42,
    "n_jobs": -1,
}


def train(
    X_train: Union[pd.Series, List[str]],
    y_train: Union[pd.Series, List[int]],
    **kwargs,
) -> RandomForestClassifier:
    """Train a Random Forest classifier on URL features.

    Parameters
    ----------
    X_train:
        URLs for training.
    y_train:
        Binary labels (0 = legitimate, 1 = phishing).
    **kwargs:
        Additional keyword arguments forwarded to
        :class:`~sklearn.ensemble.RandomForestClassifier`.

    Returns
    -------
    RandomForestClassifier
        Fitted classifier.
    """
    params = {**_DEFAULT_PARAMS, **kwargs}
    logger.info("Training RandomForestClassifier with params: %s", params)

    X_feat = build_feature_matrix(list(X_train))
    clf = RandomForestClassifier(**params)
    clf.fit(X_feat, list(y_train))
    logger.info("Training complete (%d samples)", len(X_feat))
    return clf


def save_model(clf: RandomForestClassifier, path: str) -> None:
    """Persist *clf* to disk using joblib.

    Parameters
    ----------
    clf:
        Fitted classifier to save.
    path:
        Destination file path (e.g. ``models/trained_model.joblib``).
    """
    dest = Path(path)
    dest.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(clf, dest)
    logger.info("Model saved to %s", dest)


def load_model(path: str) -> RandomForestClassifier:
    """Load a persisted classifier from disk.

    Parameters
    ----------
    path:
        Path to the joblib file.

    Raises
    ------
    FileNotFoundError
        If *path* does not exist.
    """
    model_path = Path(path)
    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")
    clf = joblib.load(model_path)
    logger.info("Model loaded from %s", model_path)
    return clf


def predict(
    clf: RandomForestClassifier,
    urls: Union[List[str], str],
) -> np.ndarray:
    """Return class predictions (0 or 1) for *urls*.

    Parameters
    ----------
    clf:
        Fitted classifier.
    urls:
        Single URL string or list of URL strings.
    """
    if isinstance(urls, str):
        urls = [urls]
    X_feat = build_feature_matrix(urls)
    return clf.predict(X_feat)


def predict_proba(
    clf: RandomForestClassifier,
    urls: Union[List[str], str],
) -> np.ndarray:
    """Return phishing probability scores for *urls*.

    Parameters
    ----------
    clf:
        Fitted classifier.
    urls:
        Single URL string or list of URL strings.

    Returns
    -------
    np.ndarray
        Array of shape ``(n_samples, 2)`` with class probabilities.
    """
    if isinstance(urls, str):
        urls = [urls]
    X_feat = build_feature_matrix(urls)
    return clf.predict_proba(X_feat)
