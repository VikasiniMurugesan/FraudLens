from pathlib import Path

# --- Paths ---------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent
DATA_DIR = PROJECT_ROOT / "data"
MODELS_DIR = PROJECT_ROOT / "models"

SAMPLE_CSV_PATH = DATA_DIR / "sample_transactions.csv"
PUBLIC_DATASET_PATH = DATA_DIR / "creditcard.csv"  # downloaded by you, not committed
MODEL_PATH = MODELS_DIR / "fraud_model.joblib"
METRICS_PATH = MODELS_DIR / "metrics.json"

# --- Reproducibility -----------------------------------------------------
RANDOM_SEED = 42

# --- Risk categories -----------------------------------------------------
# Risk score is 0-100. These are PLACEHOLDER cut-offs. We will choose real
# values later from validation data, based on false-positive/false-negative
# costs and how many alerts analysts can realistically review.
RISK_THRESHOLDS = {
    "medium": 30,  # score >= 30 -> medium
    "high": 70,    # score >= 70 -> high
}