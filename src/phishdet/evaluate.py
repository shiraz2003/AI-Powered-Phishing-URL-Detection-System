"""Model evaluation: metrics and reports."""

from typing import List, Union

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from phishdet.utils import get_logger

logger = get_logger(__name__)


def evaluate(
    y_true: Union[List[int], np.ndarray],
    y_pred: Union[List[int], np.ndarray],
    y_proba: Union[List[float], np.ndarray, None] = None,
) -> dict:
    """Compute a suite of classification metrics.

    Parameters
    ----------
    y_true:
        Ground-truth binary labels.
    y_pred:
        Predicted binary labels.
    y_proba:
        Predicted probabilities for the positive class (optional, used for
        AUC-ROC).

    Returns
    -------
    dict
        Dictionary containing ``accuracy``, ``precision``, ``recall``,
        ``f1``, and optionally ``roc_auc``.
    """
    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
    }

    if y_proba is not None:
        proba = np.asarray(y_proba)
        if proba.ndim == 2:
            proba = proba[:, 1]
        metrics["roc_auc"] = roc_auc_score(y_true, proba)

    logger.info("Evaluation results: %s", metrics)
    return metrics


def print_report(
    y_true: Union[List[int], np.ndarray],
    y_pred: Union[List[int], np.ndarray],
) -> None:
    """Print a full sklearn classification report plus confusion matrix.

    Parameters
    ----------
    y_true:
        Ground-truth binary labels.
    y_pred:
        Predicted binary labels.
    """
    target_names = ["Legitimate (0)", "Phishing (1)"]
    print("\n=== Classification Report ===")
    print(classification_report(y_true, y_pred, target_names=target_names))
    print("=== Confusion Matrix ===")
    cm = confusion_matrix(y_true, y_pred)
    cm_df = pd.DataFrame(
        cm,
        index=[f"Actual {t}" for t in target_names],
        columns=[f"Predicted {t}" for t in target_names],
    )
    print(cm_df)
    print()
