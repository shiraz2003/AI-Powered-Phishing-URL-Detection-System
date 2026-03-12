"""Model explainability via SHAP values and built-in feature importances."""

from typing import List, Union

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

from phishdet.features import FEATURE_NAMES, build_feature_matrix
from phishdet.utils import get_logger

logger = get_logger(__name__)


def feature_importances(clf: RandomForestClassifier) -> pd.Series:
    """Return a sorted Series of feature importances from the fitted *clf*.

    Parameters
    ----------
    clf:
        A fitted :class:`~sklearn.ensemble.RandomForestClassifier`.

    Returns
    -------
    pd.Series
        Feature importances indexed by feature name, sorted descending.
    """
    importances = pd.Series(clf.feature_importances_, index=FEATURE_NAMES)
    return importances.sort_values(ascending=False)


def shap_explain(
    clf: RandomForestClassifier,
    urls: Union[List[str], str],
    *,
    max_display: int = 10,
) -> None:
    """Print a text summary of SHAP values for *urls*.

    Requires the ``shap`` package to be installed.

    Parameters
    ----------
    clf:
        A fitted :class:`~sklearn.ensemble.RandomForestClassifier`.
    urls:
        Single URL string or list of URL strings to explain.
    max_display:
        Maximum number of features to display in the summary.
    """
    try:
        import shap  # noqa: PLC0415
    except ImportError as exc:
        raise ImportError(
            "The 'shap' package is required for SHAP explanations. "
            "Install it with: pip install shap"
        ) from exc

    if isinstance(urls, str):
        urls = [urls]

    X_feat = build_feature_matrix(urls)
    explainer = shap.TreeExplainer(clf)
    shap_values = explainer.shap_values(X_feat)

    # shap_values is a list [class_0, class_1] for binary classifiers
    positive_class_shap = np.array(shap_values[1] if isinstance(shap_values, list) else shap_values)

    mean_abs_shap = pd.Series(
        np.abs(positive_class_shap).mean(axis=0),
        index=FEATURE_NAMES,
    ).sort_values(ascending=False)

    print(f"\n=== SHAP Feature Importance (top {max_display}) ===")
    print(mean_abs_shap.head(max_display).to_string())
    print()
