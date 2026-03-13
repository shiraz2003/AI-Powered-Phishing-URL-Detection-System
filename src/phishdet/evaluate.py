from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

def evaluate_model(
    model: ClassifierMixin,
    X,
    y: np.ndarray,
    name: str = "Eval",
) -> Dict[str, float]:
    """Compute metrics and log them."""
    y_pred = model.predict(X)
    y_proba = None
    if hasattr(model, "predict_proba"):
        y_proba = model.predict_proba(X)[:, 1]

    acc = accuracy_score(y, y_pred)
    prec = precision_score(y, y_pred, zero_division=0)
    rec = recall_score(y, y_pred, zero_division=0)
    f1 = f1_score(y, y_pred, zero_division=0)
    if y_proba is not None:
        roc = roc_auc_score(y, y_proba)
    else:
        roc = float("nan")
    cm = confusion_matrix(y, y_pred)

    logger.info(
        f"{name} Metrics:\n"
        f"  Accuracy:  {acc:.4f}\n"
        f"  Precision: {prec:.4f}\n"
        f"  Recall:    {rec:.4f}\n"
        f"  F1 Score:  {f1:.4f}\n"
        f"  ROC AUC:   {roc:.4f}\n"
        f"  Confusion Matrix:\n{cm}\n"
    )

    return {
        "accuracy": float(acc),
        "precision": float(prec),
        "recall": float(rec),
        "f1": float(f1),
        "roc_auc": float(roc),
    }
