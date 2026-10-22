# FraudLens

An ML-based fraud detection project that scores card transactions from 0 to 100
and flags them for analyst review. **Status: in progress.**

## What works so far
- Validated data loading with duplicate removal
- Time-based train/validation/test split (no future data leaks into training)
- Logistic Regression baseline and Random Forest, compared on precision, recall and PR-AUC
- Risk score (0 to 100) with low/medium/high categories
- Streamlit app: score the held-out test set or an uploaded CSV, view flagged transactions and a single transaction's details
- Automated tests with pytest

## Not built yet
Explanations, model evaluation page, alerts workflow, deployment.

## Run it
```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python -m src.train_model
streamlit run app.py
```

## Data
Uses the public Credit Card Fraud Detection dataset from Kaggle. The file is **not**
included in this repository; see `data/README.md` for where to get it and its limitations.
Scores are rankings, not calibrated probabilities.