from __future__ import annotations

import click
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Optional

from .data import load_dataset, make_synthetic_dataset
from .model import train_model, predict_url, load_model, MODEL_FILENAME

# Simple logging replacement
import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@click.group()
def cli() -> None:
    """Phishing URL detector CLI."""
    pass

@cli.command()
@click.option(
    "--data",
    "data_path",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    default=None,
    help="Path to CSV dataset with columns url,label",
)
@click.option(
    "--synthetic-size",
    type=int,
    default=2000,
    show_default=True,
    help="Size of synthetic dataset",
)
def train(data_path: Optional[Path], synthetic_size: int) -> None:
    """Train phishing URL detection model."""
    if data_path is None:
        logger.info("No data path provided. Generating synthetic dataset...")
        df = make_synthetic_dataset(n=synthetic_size)
    else:
        logger.info(f"Loading dataset from {data_path}")
        df = load_dataset(data_path)

    logger.info(f"Dataset shape: {df.shape}")
    logger.info(f"Label distribution:\n{df['label'].value_counts()}")

    metadata = train_model(df)
    logger.info("✅ Training complete!")
    logger.info(f"Model saved. Test accuracy: {metadata['test_accuracy']:.3f}")

@cli.command()
@click.option("--url", "url_str", type=str, required=True, help="URL to classify")
def predict(url_str: str) -> None:
    """Predict phishing/benign for a single URL."""
    try:
        label, proba = predict_url(url_str)
        label_str = "🟥 PHISHING" if label == 1 else "✅ BENIGN"
        click.echo(f"URL: {url_str}")
        click.echo(f"Prediction: {label_str}")
        click.echo(f"Probability: {proba:.4f}")
    except Exception as e:
        click.echo(f"❌ Error: {e}")
        click.echo("Run 'train' command first.")

@cli.command(name="predict-file")
@click.option(
    "--input",
    "input_path",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    required=True,
    help="Input CSV file with column 'url'",
)
@click.option(
    "--output",
    "output_path", 
    type=click.Path(dir_okay=False, path_type=Path),
    required=True,
    help="Output CSV with predictions",
)
def predict_file(input_path: Path, output_path: Path) -> None:
    """Predict phishing/benign for batch of URLs."""
    try:
        clf, vectorizer = load_model()
        print(f"✅ Loaded model")
        
        # Load input CSV
        df = pd.read_csv(input_path)
        print(f"📊 Processing {len(df)} URLs from {input_path}")
        
        # Assume first column is URLs, or find 'url' column
        if 'url' in df.columns:
            urls = df['url'].tolist()
        else:
            urls = df.iloc[:, 0].astype(str).tolist()  # First column
        
        # Predict all URLs
        X = vectorizer.transform(urls)
        probas = clf.predict_proba(X)[:, 1]  # Phishing probability
        labels = (probas >= 0.5).astype(int)
        
        # Create results DataFrame
        results_df = pd.DataFrame({
            'url': urls,
            'prediction': labels,
            'phishing_probability': probas,
            'status': ['🟥 PHISHING' if l == 1 else '✅ BENIGN' for l in labels]
        })
        
        # Save results
        results_df.to_csv(output_path, index=False)
        print(f"💾 Results saved to {output_path}")
        
        # Show summary
        phishing_count = sum(labels)
        print(f"\n📈 RESULTS SUMMARY:")
        print(f"Total URLs: {len(urls)}")
        print(f"Phishing: {phishing_count} ({phishing_count/len(urls)*100:.1f}%)")
        print(f"Benign: {len(urls)-phishing_count}")
        print("\n🔗 First 5 predictions:")
        print(results_df.to_string(index=False))
        
    except Exception as e:
        print(f"❌ Error: {e}")
        print("Make sure model is trained first!")

if __name__ == "__main__":
    cli()
