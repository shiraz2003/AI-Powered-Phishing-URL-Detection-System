import pandas as pd
import numpy as np
from pathlib import Path

def load_dataset(path: Path) -> pd.DataFrame:
    """Load dataset from CSV with label flipping."""
    df = pd.read_csv(path)
    print(f"CSV columns found: {list(df.columns)}")
    
    # Find URL column
    url_col = None
    for col in df.columns:
        if 'url' in str(col).lower():
            url_col = col
            break
    if url_col is None:
        url_col = df.columns[1]  # Usually second column
    
    # Find label column  
    label_col = None
    for col in df.columns:
        if 'label' in str(col).lower():
            label_col = col
            break
    if label_col is None:
        label_col = 'label'  # Default
    
    print(f"Using URL column: '{url_col}'")
    print(f"Using label column: '{label_col}'")
    
    # Create clean dataframe
    df_clean = pd.DataFrame({
        'url': df[url_col].astype(str).fillna('').tolist(),
        'label': pd.to_numeric(df[label_col], errors='coerce').fillna(0).astype(int)
    })
    
    # FLIP LABELS: PhiUSIIL uses 1=legit, 0=phish → We want 0=benign, 1=phish
    df_clean['label'] = 1 - df_clean['label']
    
    # Take first 2000 rows for speed
    df_clean = df_clean.head(2000)
    print(f"Loaded {len(df_clean)} rows")
    print(f"Labels (0=benign, 1=phishing): {df_clean['label'].value_counts().to_dict()}")
    
    return df_clean

def make_synthetic_dataset(n: int = 2000) -> pd.DataFrame:
    """Create synthetic dataset."""
    np.random.seed(42)
    benign_urls = ["https://www.google.com"] * (n//2)
    phishing_urls = ["http://fake-login.com"] * (n//2)
    urls = benign_urls + phishing_urls
    np.random.shuffle(urls)
    
    labels = [0] * (n//2) + [1] * (n//2)  # 0=benign, 1=phishing
    df = pd.DataFrame({'url': urls, 'label': labels})
    print(f"Created synthetic dataset: {len(df)} rows")
    return df
