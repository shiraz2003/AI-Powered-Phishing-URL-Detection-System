"""Command-line interface for the phishing URL detection system."""

import argparse
import sys
from pathlib import Path

from phishdet.data import load_data, split_data
from phishdet.evaluate import evaluate, print_report
from phishdet.explain import feature_importances, shap_explain
from phishdet.model import load_model, predict, predict_proba, save_model, train
from phishdet.utils import get_logger, validate_url

logger = get_logger(__name__)

_DEFAULT_DATA = str(Path(__file__).parent.parent.parent / "data" / "sample_urls.csv")
_DEFAULT_MODEL = str(Path(__file__).parent.parent.parent / "models" / "trained_model.joblib")


# ---------------------------------------------------------------------------
# Subcommand handlers
# ---------------------------------------------------------------------------

def _cmd_train(args: argparse.Namespace) -> int:
    df = load_data(args.data)
    X_train, X_test, y_train, y_test = split_data(
        df, test_size=args.test_size, random_state=args.seed
    )
    clf = train(X_train, y_train, n_estimators=args.estimators, random_state=args.seed)
    save_model(clf, args.model)

    y_pred = predict(clf, list(X_test))
    y_proba = predict_proba(clf, list(X_test))
    metrics = evaluate(list(y_test), y_pred, y_proba)
    print_report(list(y_test), y_pred)

    print("=== Metrics ===")
    for name, value in metrics.items():
        print(f"  {name:12s}: {value:.4f}")

    print("\n=== Feature Importances ===")
    print(feature_importances(clf).to_string())
    return 0


def _cmd_predict(args: argparse.Namespace) -> int:
    clf = load_model(args.model)

    urls: list[str] = []
    if args.url:
        urls = [args.url]
    elif args.file:
        with open(args.file) as fh:
            urls = [line.strip() for line in fh if line.strip()]
    else:
        print("Error: provide --url or --file.", file=sys.stderr)
        return 1

    invalid = [u for u in urls if not validate_url(u)]
    if invalid:
        print(f"Warning: {len(invalid)} URL(s) with unrecognised scheme skipped.", file=sys.stderr)
        urls = [u for u in urls if validate_url(u)]

    if not urls:
        print("No valid URLs to classify.", file=sys.stderr)
        return 1

    preds = predict(clf, urls)
    probas = predict_proba(clf, urls)[:, 1]

    print(f"\n{'URL':<60} {'Label':<10} {'Phishing Prob':>13}")
    print("-" * 85)
    for url, pred, prob in zip(urls, preds, probas):
        label = "PHISHING" if pred == 1 else "LEGITIMATE"
        print(f"{url:<60} {label:<10} {prob:>13.4f}")
    return 0


def _cmd_explain(args: argparse.Namespace) -> int:
    clf = load_model(args.model)
    if args.url:
        shap_explain(clf, args.url, max_display=args.top)
    else:
        print("=== Feature Importances (Random Forest) ===")
        print(feature_importances(clf).head(args.top).to_string())
    return 0


# ---------------------------------------------------------------------------
# Parser construction
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="phishdet",
        description="AI-Powered Phishing URL Detection System",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # train
    p_train = sub.add_parser("train", help="Train the classifier and save the model.")
    p_train.add_argument("--data", default=_DEFAULT_DATA, help="Path to the CSV dataset.")
    p_train.add_argument("--model", default=_DEFAULT_MODEL, help="Output path for the saved model.")
    p_train.add_argument("--test-size", type=float, default=0.2, dest="test_size")
    p_train.add_argument("--estimators", type=int, default=100)
    p_train.add_argument("--seed", type=int, default=42)
    p_train.set_defaults(func=_cmd_train)

    # predict
    p_predict = sub.add_parser("predict", help="Classify one or more URLs.")
    p_predict.add_argument("--model", default=_DEFAULT_MODEL, help="Path to the saved model.")
    group = p_predict.add_mutually_exclusive_group()
    group.add_argument("--url", help="Single URL to classify.")
    group.add_argument("--file", help="Path to a text file with one URL per line.")
    p_predict.set_defaults(func=_cmd_predict)

    # explain
    p_explain = sub.add_parser("explain", help="Show feature importances or SHAP values.")
    p_explain.add_argument("--model", default=_DEFAULT_MODEL, help="Path to the saved model.")
    p_explain.add_argument("--url", help="URL to compute SHAP values for (requires shap package).")
    p_explain.add_argument("--top", type=int, default=10, help="Number of top features to display.")
    p_explain.set_defaults(func=_cmd_explain)

    return parser


def main(argv=None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
