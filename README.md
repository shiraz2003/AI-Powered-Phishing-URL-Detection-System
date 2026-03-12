# AI-Powered Phishing URL Detection System

A machine-learning pipeline that classifies URLs as **legitimate** or
**phishing** using hand-crafted lexical features and a Random Forest
classifier, with optional SHAP-based explanations.

---

## Repository structure

```
AI-Powered-Phishing-URL-Detection-System-/
│
├── data/
│   └── sample_urls.csv          # Labelled URL dataset (0 = legit, 1 = phish)
│
├── models/
│   └── trained_model.joblib     # Serialised trained classifier
│
├── src/
│   └── phishdet/
│       ├── __init__.py
│       ├── data.py              # CSV loading & train/test split
│       ├── preprocess.py        # URL normalisation helpers
│       ├── features.py          # Lexical feature extraction
│       ├── model.py             # Train / save / load / predict
│       ├── evaluate.py          # Metrics & classification report
│       ├── explain.py           # Feature importances & SHAP values
│       ├── cli.py               # argparse command-line interface
│       └── utils.py             # Logging & miscellaneous helpers
│
├── tests/
│   ├── test_preprocess.py
│   └── test_model.py
│
├── requirements.txt
├── README.md
└── run.sh
```

---

## Quick start

### 1 – Install dependencies

```bash
pip install -r requirements.txt
```

### 2 – Train the model

```bash
./run.sh train
# or: python -m phishdet.cli train  (with src/ on PYTHONPATH)
```

Options:

| Flag | Default | Description |
|------|---------|-------------|
| `--data` | `data/sample_urls.csv` | Path to the CSV dataset |
| `--model` | `models/trained_model.joblib` | Output path for the saved model |
| `--test-size` | `0.2` | Fraction of data reserved for evaluation |
| `--estimators` | `100` | Number of trees in the Random Forest |
| `--seed` | `42` | Random seed |

### 3 – Classify a URL

```bash
./run.sh predict --url "http://paypal.com-secure.evil.com/login"
```

Or batch-classify URLs from a file (one URL per line):

```bash
./run.sh predict --file my_urls.txt
```

### 4 – Explain predictions

Print Random Forest feature importances:

```bash
./run.sh explain --top 10
```

Print SHAP values for a specific URL (requires `shap` to be installed):

```bash
./run.sh explain --url "http://192.168.1.1/phishing"
```

---

## Features extracted from each URL

| Feature | Description |
|---------|-------------|
| `url_length` | Total character length |
| `dot_count` | Number of `.` characters |
| `hyphen_count` | Number of `-` characters |
| `at_count` | Number of `@` characters |
| `slash_count` | Number of `/` characters |
| `question_count` | Number of `?` characters |
| `equals_count` | Number of `=` characters |
| `underscore_count` | Number of `_` characters |
| `digit_count` | Number of digit characters |
| `special_char_count` | Total non-alphanumeric characters |
| `has_https` | 1 if scheme is HTTPS |
| `has_ip` | 1 if netloc contains a bare IP address |
| `subdomain_count` | Number of subdomains (excluding `www`) |
| `domain_length` | Character length of the hostname |
| `path_length` | Character length of the URL path |
| `has_query` | 1 if a query string is present |
| `is_shortener` | 1 if the domain is a known URL shortener |
| `double_slash_redirect` | 1 if `//` appears in the path |
| `suspicious_keyword_count` | Count of phishing-related keywords |

---

## Running tests

```bash
PYTHONPATH=src pytest tests/ -v
```

---

## Requirements

See [`requirements.txt`](requirements.txt).  Core dependencies:

* `scikit-learn` – Random Forest classifier
* `pandas` / `numpy` – data handling
* `joblib` – model serialisation
* `shap` *(optional)* – SHAP-based explanations
