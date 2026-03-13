import shap

from .model import load_model
from .utils import get_models_dir

def _get_feature_names(feature_builder: FeatureBuilder) -> List[str]:
    """Return combined feature names: TF-IDF + numeric."""
    tfidf_names = list(feature_builder.vectorizer.get_feature_names_out())
    numeric_names = [
        "url_length",
        "hostname_length",
        "num_digits",
        "num_dots",
        "num_hyphens",
        "num_special_chars",
        "num_subdomains",
        "path_depth",
        "has_ip_address",
        "suspicious_keyword_count",
    ]
    return tfidf_names + numeric_names
def compute_feature_importance(
    sample_size: int = 200,
) -> Tuple[pd.DataFrame, shap.Explainer]:
    """Compute SHAP feature importances on a small synthetic subset."""
    clf, feature_builder = load_model()
    models_dir = get_models_dir()
    data_path = models_dir.parent / "data" / "sample_urls.csv"
    if not data_path.exists():
        raise FileNotFoundError(
            f"Dataset not found at {data_path}; required for SHAP."
        )

    df = pd.read_csv(data_path)
    if len(df) > sample_size:
        df_sample = df.sample(n=sample_size, random_state=0)
    else:
        df_sample = df

    X_sample = feature_builder.transform(df_sample["url"].tolist())

    # Use LinearExplainer for LogisticRegression
    explainer = shap.LinearExplainer(clf, X_sample, feature_perturbation="interventional")
    shap_values = explainer(X_sample)

    feature_names = _get_feature_names(feature_builder)
    mean_abs_shap = np.mean(np.abs(shap_values.values), axis=0)

    importance_df = pd.DataFrame(
        {"feature": feature_names, "mean_abs_shap": mean_abs_shap}
    )
    importance_df = importance_df.sort_values(
        "mean_abs_shap", ascending=False
    ).reset_index(drop=True)

    out_path = models_dir / "feature_importance_shap.csv"
    importance_df.to_csv(out_path, index=False)
    logger.info(
        f"Saved SHAP feature importance (top 20) to {out_path}\n"
    )
    return importance_df.head(20), explainer
