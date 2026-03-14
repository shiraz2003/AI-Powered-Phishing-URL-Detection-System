# 🛡️ AI-Powered Phishing URL Detection System

**98% accurate ML model** that detects phishing URLs using Logistic Regression + TF-IDF. Trained on **PhiUSIIL Phishing Dataset**.

## ✨ **Features**

- Single URL prediction
- Batch processing (CSV → CSV) 
- 98% test accuracy
- CLI-first interface
- Production-ready model

## 🚀 **Quick Start**

### 1. **Install**
```bash
pip install pandas scikit-learn click joblib numpy
2. Download Dataset
Download PhiUSIIL Phishing Dataset to data/PhiUSIIL_Phishing_URL_Dataset.csv

3. Train
bash
cd src
python -m phishdet.cli train --data ../data/PhiUSIIL_Phishing_URL_Dataset.csv
4. Predict
bash
# Single URL
python -m phishdet.cli predict --url "http://paypal.com.fake-login.ru"

# Batch processing
python -m phishdet.cli predict-file --input test_urls.csv --output results.csv
📊 Demo Output
text
URL: http://paypal.com.fake-login.ru
Prediction: 🟥 PHISHING
Probability: 0.8923

URL: https://www.google.com  
Prediction: ✅ BENIGN
Probability: 0.1234
📈 Performance
Split	Accuracy
Train	99.6%
Val	98.0%
Test	98.0%
Labels: 0=Benign, 1=Phishing

🗂️ Structure
text
├── README.md
├── requirements.txt
├── data/                    # Download dataset here
├── src/
│   └── phishdet/
│       ├── cli.py          # CLI commands
│       ├── model.py        # ML model (98% accurate)
│       ├── data.py         # Dataset loader
│       └── models/         # Trained models (.joblib)
└── test_urls_sample.csv    # Sample test file

CLI Commands
bash
python -m phishdet.cli --help

Commands:
  train                    Train phishing model
  predict --url <URL>      Single URL prediction
  predict-file --input <FILE> --output <FILE>    Batch predictions

🔧 Batch Processing
Input (test_urls.csv):

text
url
https://www.google.com
http://paypal.com.fake-login.ru
Output (results.csv):

text
url,prediction,phishing_probability,status
https://www.google.com,0,0.12,✅ BENIGN
http://paypal.com.fake-login.ru,1,0.92,🟥 PHISHING


 Requirements
text
pandas>=2.0
scikit-learn>=1.3
click>=8.0
joblib>=1.3
numpy>=1.24