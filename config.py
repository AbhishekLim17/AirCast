"""
config.py — Single source of truth for all project constants.
All values that might need tuning live here. Never scatter magic numbers across files.
"""

import os
from datetime import date as _date, datetime as _datetime
from zoneinfo import ZoneInfo

from dotenv import load_dotenv

load_dotenv()

# ─── Timezone ──────────────────────────────────────────────────────────────────
# Ahmedabad is IST (UTC+5:30).  All date logic must use IST to avoid
# mismatches between GitHub Actions (UTC), Streamlit Cloud, and the WAQI API.
IST = ZoneInfo("Asia/Kolkata")


def today_ist() -> _date:
    """Return today's date in IST regardless of the server's system timezone."""
    return _datetime.now(IST).date()


def now_ist() -> _datetime:
    """Return the current datetime in IST."""
    return _datetime.now(IST)

# ─── WAQI API ─────────────────────────────────────────────────────────────────
WAQI_API_TOKEN: str = os.getenv("WAQI_API_TOKEN", "")
WAQI_BASE_URL: str = "https://api.waqi.info"

# Ahmedabad monitoring stations — confirmed working slugs only.
# To find more: visit https://aqicn.org/city/india/ahmedabad/ and note the URL slug.
AHMEDABAD_STATIONS: list[str] = [
    "ahmedabad",            # AUDA composite station — confirmed working
]

# Primary station used for single-station DB queries and predictions.
PRIMARY_STATION: str = "ahmedabad"

# ─── Supabase ─────────────────────────────────────────────────────────────────
SUPABASE_URL: str = os.getenv("SUPABASE_URL", "")
SUPABASE_KEY: str = os.getenv("SUPABASE_KEY", "")

# ─── Hugging Face Hub ─────────────────────────────────────────────────────────
HF_TOKEN: str = os.getenv("HF_TOKEN", "")
HF_USERNAME: str = os.getenv("HF_USERNAME", "")
HF_REPO_NAME: str = os.getenv("HF_REPO_NAME", "ahmedabad-aqi-model")
HF_MODEL_FILENAME: str = "xgb_model.pkl"

HF_REPO_ID: str = f"{HF_USERNAME}/{HF_REPO_NAME}"

# ─── Model & Retraining ───────────────────────────────────────────────────────
# MAE (Mean Absolute Error) threshold that triggers automatic retraining.
RETRAIN_MAE_THRESHOLD: float = float(os.getenv("RETRAIN_MAE_THRESHOLD", "12"))

# Number of historical days used for retraining window.
# Must be larger than RETRAIN_TEST_DAYS to leave data for training.
RETRAIN_WINDOW_DAYS: int = 365

# Days reserved for the test set during temporal split.
RETRAIN_TEST_DAYS: int = 60

# Walk-forward CV folds for time-series validation.
CV_FOLDS: int = 8

# Optuna hyperparameter tuning trials.
OPTUNA_TRIALS: int = 150

# ─── Feature Engineering ──────────────────────────────────────────────────────
# Lag offsets in days to create as features.
LAG_DAYS: list[int] = [1, 2, 3, 5, 7, 14, 21, 30]

# Rolling window sizes in days.
ROLLING_WINDOWS: list[int] = [3, 7, 14, 30]

# ─── AQI Health Categories (CPCB India standard) ──────────────────────────────
AQI_CATEGORIES: list[dict] = [
    {"label": "Good",           "min": 0,   "max": 50,  "color": "#16a34a"},
    {"label": "Satisfactory",   "min": 51,  "max": 100, "color": "#65a30d"},
    {"label": "Moderate",       "min": 101, "max": 200, "color": "#d97706"},
    {"label": "Poor",           "min": 201, "max": 300, "color": "#ea580c"},
    {"label": "Very Poor",      "min": 301, "max": 400, "color": "#dc2626"},
    {"label": "Severe",         "min": 401, "max": 500, "color": "#7f1d1d"},
]

def get_aqi_category(aqi: float) -> dict:
    """Return the AQI category dict for a given AQI value. Clamps invalid AQI to valid range.

    Uses each category's max boundary so float values like 50.5 are
    classified as 'Satisfactory' (not 'Good').
    """
    # Clamp AQI to valid range [0, 500] to handle edge cases
    aqi = max(0.0, min(float(aqi), 500.0))

    for cat in AQI_CATEGORIES:
        if aqi <= cat["max"]:
            return cat

    return AQI_CATEGORIES[-1]  # Unreachable, but keeps type-checkers happy

# ─── Paths ────────────────────────────────────────────────────────────────────
import pathlib

ROOT_DIR = pathlib.Path(__file__).parent
DATA_RAW_DIR = ROOT_DIR / "data" / "raw"
DATA_PROCESSED_DIR = ROOT_DIR / "data" / "processed"
PROCESSED_CSV = DATA_PROCESSED_DIR / "ahmedabad_clean.csv"


# ─── Config validation ────────────────────────────────────────────────────────

def validate_config() -> None:
    """Raise EnvironmentError early if any required env vars are missing.

    Call this at the top of daily_job.py and any entry-point script so the
    process fails fast with a clear message instead of making real HTTP
    requests with empty tokens.
    """
    required = {
        "WAQI_API_TOKEN": WAQI_API_TOKEN,
        "SUPABASE_URL":   SUPABASE_URL,
        "SUPABASE_KEY":   SUPABASE_KEY,
        "HF_TOKEN":       HF_TOKEN,
        "HF_USERNAME":    HF_USERNAME,
    }
    missing = [name for name, val in required.items() if not val]
    if missing:
        raise EnvironmentError(
            f"Required environment variables are not set: {', '.join(missing)}\n"
            "Copy .env.example to .env and fill in the values."
        )
