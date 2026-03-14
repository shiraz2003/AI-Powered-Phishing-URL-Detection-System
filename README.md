# 🛡️ AI-Powered Phishing URL Detection System

A **machine learning–based phishing URL detection** project that classifies URLs as **Benign** or **Phishing** using **TF-IDF vectorization + Logistic Regression**.

The model is trained on the **PhiUSIIL Phishing URL Dataset** and achieves ~**98% accuracy** on the test set. The project includes a **CLI tool** to predict phishing URLs individually or in batches (CSV).

---

## ✨ Features

- ✅ Detect phishing URLs with ~98% accuracy  
- ✅ Single URL prediction from command line  
- ✅ Batch prediction using CSV files  
- ✅ Fast TF-IDF feature extraction  
- ✅ Production-ready model saved with **Joblib**  
- ✅ Simple CLI interface using **Click**

---

## 📊 Model Performance

| Dataset     | Accuracy |
|------------|----------|
| Train      | 99.6%    |
| Validation | 98.0%    |
| Test       | 98.0%    |

---

## 🏷️ Label Encoding

- `0` = Benign URL  
- `1` = Phishing URL  

---

## 📦 Project Structure

```text
AI-Powered-Phishing-URL-Detection-System/
│
├── README.md
├── requirements.txt
│
├── data/
│   └── PhiUSIIL_Phishing_URL_Dataset.csv
│
├── src/
│   └── phishdet/
│       ├── cli.py
│       ├── model.py
│       ├── data.py
│       └── models/
│           └── phishing_model.joblib
│
└── test_urls_sample.csv
```

---

## ⚙️ Installation

### 1) Clone the repository

```bash
git clone https://github.com/shiraz2003/AI-Powered-Phishing-URL-Detection-System.git
cd AI-Powered-Phishing-URL-Detection-System
```

### 2) Install dependencies

**Recommended:**
```bash
pip install -r requirements.txt
```

Or install manually:
```bash
pip install pandas scikit-learn click joblib numpy
```

---

## 📥 Dataset

Download the **PhiUSIIL Phishing URL Dataset** and place it here:

```text
data/PhiUSIIL_Phishing_URL_Dataset.csv
```

Dataset source:
- https://archive.ics.uci.edu/ml/index.php

---

## 🚀 Training the Model

From the repository root:

```bash
cd src
python -m phishdet.cli train --data ../data/PhiUSIIL_Phishing_URL_Dataset.csv
```

This will:
- Load the dataset  
- Train the **TF-IDF + Logistic Regression** model  
- Save the trained model to:

```text
src/phishdet/models/
```

---

## 🔍 Predicting URLs

### ✅ Single URL prediction

```bash
python -m phishdet.cli predict --url "http://paypal.com.fake-login.ru"
```

Example output:
```text
URL: http://paypal.com.fake-login.ru
Prediction: PHISHING
Probability: 0.8923
```

### ✅ Batch URL prediction (CSV)

Input file example (`test_urls.csv`):
```csv
url
https://www.google.com
http://paypal.com.fake-login.ru
```

Run:
```bash
python -m phishdet.cli predict-file --input test_urls.csv --output results.csv
```

Output file example (`results.csv`):
```csv
url,prediction,phishing_probability,status
https://www.google.com,0,0.12,BENIGN
http://paypal.com.fake-login.ru,1,0.92,PHISHING
```

---

## 🖥 CLI Commands

Show help:
```bash
python -m phishdet.cli --help
```

Available commands:
- `train` — Train phishing detection model  
- `predict --url <URL>` — Predict a single URL  
- `predict-file` — Batch prediction using CSV  

---

## 🧠 Model Architecture

The detection pipeline uses:

1. **TF-IDF Vectorization**  
   Converts URLs into numerical features based on character/token patterns.

2. **Logistic Regression**  
   A fast, interpretable classifier that performs well on text-like features.

Pipeline:

```text
URL → TF-IDF Vectorizer → Logistic Regression → Prediction
```

---

## 🧪 Sample Test File

A sample CSV is included:
- `test_urls_sample.csv`

You can edit it to test your own URLs.

---

## 📌 Notes / Disclaimer

This project is for educational and research purposes. Always use multiple security signals and safe browsing practices when evaluating suspicious links.
