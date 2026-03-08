"""
config.py — Single source of truth for all project constants.
All values that might need tuning live here. Never scatter magic numbers across files.
"""

import os
from dotenv import load_dotenv

load_dotenv()

# ─── WAQI API ─────────────────────────────────────────────────────────────────
WAQI_API_TOKEN: str = os.environ["WAQI_API_TOKEN"]
WAQI_BASE_URL: str = "https://api.waqi.info"

# Ahmedabad monitoring stations — confirmed working slugs only.
# To find more: visit https://aqicn.org/city/india/ahmedabad/ and note the URL slug.
AHMEDABAD_STATIONS: list[str] = [
    "ahmedabad",            # AUDA composite station — confirmed working
]

# Primary station used for single-station DB queries and predictions.
PRIMARY_STATION: str = "ahmedabad"

# ─── Supabase ─────────────────────────────────────────────────────────────────
SUPABASE_URL: str = os.environ["SUPABASE_URL"]
SUPABASE_KEY: str = os.environ["SUPABASE_KEY"]

# ─── Hugging Face Hub ─────────────────────────────────────────────────────────
HF_TOKEN: str = os.environ["HF_TOKEN"]
HF_USERNAME: str = os.environ["HF_USERNAME"]
HF_REPO_NAME: str = os.getenv("HF_REPO_NAME", "ahmedabad-aqi-model")
HF_MODEL_FILENAME: str = "xgb_model.pkl"

HF_REPO_ID: str = f"{HF_USERNAME}/{HF_REPO_NAME}"

# ─── Model & Retraining ───────────────────────────────────────────────────────
# MAE (Mean Absolute Error) threshold that triggers automatic retraining.
RETRAIN_MAE_THRESHOLD: float = float(os.getenv("RETRAIN_MAE_THRESHOLD", "20"))

# Number of historical days used for retraining window.
RETRAIN_WINDOW_DAYS: int = 90

# Walk-forward CV folds for time-series validation.
CV_FOLDS: int = 5

# Optuna hyperparameter tuning trials.
OPTUNA_TRIALS: int = 50

# ─── Feature Engineering ──────────────────────────────────────────────────────
# Lag offsets in days to create as features.
LAG_DAYS: list[int] = [1, 7, 14, 30]

# Rolling window sizes in days.
ROLLING_WINDOWS: list[int] = [7, 30]

# ─── AQI Health Categories (CPCB India standard) ──────────────────────────────
AQI_CATEGORIES: list[dict] = [
    {"label": "Good",           "min": 0,   "max": 50,  "color": "#00e400"},
    {"label": "Satisfactory",   "min": 51,  "max": 100, "color": "#92d050"},
    {"label": "Moderate",       "min": 101, "max": 200, "color": "#ffff00"},
    {"label": "Poor",           "min": 201, "max": 300, "color": "#ff7e00"},
    {"label": "Very Poor",      "min": 301, "max": 400, "color": "#ff0000"},
    {"label": "Severe",         "min": 401, "max": 500, "color": "#7e0023"},
]

def get_aqi_category(aqi: float) -> dict:
    """Return the AQI category dict for a given AQI value."""
    for cat in AQI_CATEGORIES:
        if cat["min"] <= aqi <= cat["max"]:
            return cat
    return {"label": "Severe", "min": 401, "max": 500, "color": "#7e0023"}

# ─── Paths ────────────────────────────────────────────────────────────────────
import pathlib

ROOT_DIR = pathlib.Path(__file__).parent
DATA_RAW_DIR = ROOT_DIR / "data" / "raw"
DATA_PROCESSED_DIR = ROOT_DIR / "data" / "processed"
PROCESSED_CSV = DATA_PROCESSED_DIR / "ahmedabad_clean.csv"
