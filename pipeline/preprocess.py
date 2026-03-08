"""
pipeline/preprocess.py
Loads the raw Kaggle India AQI CSV, filters for Ahmedabad, engineers features,
and writes a clean training-ready CSV to data/processed/ahmedabad_clean.csv.

Kaggle dataset: "Air Quality Data in India (2015-2020)"
  → https://www.kaggle.com/datasets/rohanrao/air-quality-data-in-india
  → File used: city_day.csv

Expected raw columns (subset used):
    City, Date, PM2.5, PM10, NO, NO2, NOx, NH3, CO, SO2, O3, AQI

Usage (standalone):
    python -m pipeline.preprocess
    python -m pipeline.preprocess --input data/raw/city_day.csv
"""

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from config import (
    DATA_RAW_DIR, DATA_PROCESSED_DIR, PROCESSED_CSV,
    LAG_DAYS, ROLLING_WINDOWS,
)

logger = logging.getLogger(__name__)

# ─── Constants ────────────────────────────────────────────────────────────────

CITY_NAME = "Ahmedabad"

# Pollutant columns present in Kaggle city_day.csv
POLLUTANT_COLS = ["PM2.5", "PM10", "NO", "NO2", "NOx", "NH3", "CO", "SO2", "O3"]

# Maximum consecutive missing days to forward-fill (longer gaps get dropped)
MAX_FILL_DAYS = 3


# ─── Public entry point ───────────────────────────────────────────────────────

def build_features(raw_csv: Path = None) -> pd.DataFrame:
    """
    Full preprocessing pipeline:
      1. Load raw CSV
      2. Filter Ahmedabad, parse dates, sort
      3. Fill short gaps, drop long ones
      4. Engineer lag/rolling/cyclical features
      5. Drop rows still missing the target (AQI)
      6. Return clean DataFrame

    Args:
        raw_csv: Path to city_day.csv. Defaults to data/raw/city_day.csv.

    Returns:
        DataFrame ready for model training, indexed by date.
    """
    raw_csv = raw_csv or DATA_RAW_DIR / "city_day.csv"

    df = _load_and_filter(raw_csv)
    df = _fill_gaps(df)
    df = _engineer_features(df)
    df = _drop_incomplete_rows(df)

    logger.info("Final dataset: %d rows, %d columns", len(df), df.shape[1])
    return df


def save_processed(df: pd.DataFrame, path: Path = None) -> Path:
    """Save the processed DataFrame to CSV and return the path."""
    path = path or PROCESSED_CSV
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=True)
    logger.info("Saved processed data → %s", path)
    return path


def load_processed(path: Path = None) -> pd.DataFrame:
    """Load previously saved processed CSV. Returns empty DataFrame if missing."""
    path = path or PROCESSED_CSV
    if not path.exists():
        logger.warning("Processed CSV not found at %s — run preprocess first", path)
        return pd.DataFrame()
    df = pd.read_csv(path, index_col="Date", parse_dates=True)
    logger.info("Loaded processed data: %d rows from %s", len(df), path)
    return df


# ─── Internal steps ───────────────────────────────────────────────────────────

def _load_and_filter(raw_csv: Path) -> pd.DataFrame:
    """Load CSV, keep only Ahmedabad rows, parse dates, sort ascending."""
    if not raw_csv.exists():
        raise FileNotFoundError(
            f"Raw CSV not found: {raw_csv}\n"
            "Download from Kaggle:\n"
            "  kaggle datasets download -d rohanrao/air-quality-data-in-india\n"
            "  Unzip city_day.csv into data/raw/"
        )

    logger.info("Loading raw CSV from %s …", raw_csv)
    raw = pd.read_csv(raw_csv)

    # Validate required columns
    required = {"City", "Date", "AQI"}
    missing_cols = required - set(raw.columns)
    if missing_cols:
        raise ValueError(f"Raw CSV missing expected columns: {missing_cols}")

    df = raw[raw["City"] == CITY_NAME].copy()
    if df.empty:
        raise ValueError(f"No rows found for City='{CITY_NAME}' in {raw_csv}")

    df["Date"] = pd.to_datetime(df["Date"])
    df = df.sort_values("Date").set_index("Date")

    # Keep only the columns we actually use
    keep = [c for c in POLLUTANT_COLS if c in df.columns] + ["AQI"]
    df = df[keep]

    logger.info("Loaded %d rows for %s (%s → %s)",
                len(df), CITY_NAME, df.index.min().date(), df.index.max().date())
    return df


def _fill_gaps(df: pd.DataFrame) -> pd.DataFrame:
    """
    Forward-fill gaps up to MAX_FILL_DAYS consecutive missing values.
    Longer gaps are left as NaN and dropped later.
    """
    before = df.isna().sum().sum()
    df = df.ffill(limit=MAX_FILL_DAYS)
    after = df.isna().sum().sum()
    logger.info("Gap filling: %d NaNs → %d NaNs (limit=%d days)",
                before, after, MAX_FILL_DAYS)
    return df


def _engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add lag, rolling average, and cyclical time-encoding features."""
    df = df.copy()

    # ── Lag features (AQI n days ago) ──────────────────────────────────────
    for lag in LAG_DAYS:
        df[f"aqi_lag_{lag}d"] = df["AQI"].shift(lag)
        logger.debug("Added lag feature: aqi_lag_%dd", lag)

    # ── Rolling averages ───────────────────────────────────────────────────
    for window in ROLLING_WINDOWS:
        df[f"aqi_roll_{window}d"] = (
            df["AQI"].shift(1).rolling(window=window, min_periods=max(1, window // 2)).mean()
        )
        logger.debug("Added rolling feature: aqi_roll_%dd", window)

    # ── Pollutant ratio (PM2.5 / PM10) ────────────────────────────────────
    if "PM2.5" in df.columns and "PM10" in df.columns:
        df["pm_ratio"] = df["PM2.5"] / (df["PM10"].replace(0, np.nan))

    # ── Cyclical time encoding ─────────────────────────────────────────────
    # Encode day-of-week (0=Mon … 6=Sun) and month as sine/cosine pairs so the
    # model sees the cyclic nature (e.g. Sunday is close to Monday).
    dow = df.index.dayofweek
    month = df.index.month

    df["dow_sin"] = np.sin(2 * np.pi * dow / 7)
    df["dow_cos"] = np.cos(2 * np.pi * dow / 7)
    df["month_sin"] = np.sin(2 * np.pi * (month - 1) / 12)
    df["month_cos"] = np.cos(2 * np.pi * (month - 1) / 12)

    # ── Boolean season flags (northern India) ─────────────────────────────
    df["is_winter"] = month.isin([11, 12, 1, 2]).astype(int)
    df["is_summer"] = month.isin([3, 4, 5]).astype(int)
    df["is_monsoon"] = month.isin([6, 7, 8, 9]).astype(int)

    # ── Day of year (normalised 0–1) ───────────────────────────────────────
    df["day_of_year_norm"] = (df.index.dayofyear - 1) / 364.0

    logger.info("Engineered %d features total", df.shape[1])
    return df


def _drop_incomplete_rows(df: pd.DataFrame) -> pd.DataFrame:
    """
    Drop rows where the target (AQI) or any lag/rolling feature is still NaN.
    These are always at the start of the series before enough history exists.
    """
    before = len(df)
    # Must always have a valid AQI target
    df = df.dropna(subset=["AQI"])
    # Drop rows where lag features are missing (first N rows)
    lag_cols = [f"aqi_lag_{d}d" for d in LAG_DAYS]
    df = df.dropna(subset=lag_cols)
    after = len(df)
    logger.info("Dropped %d incomplete rows (%.1f%%) — %d rows remain",
                before - after, 100 * (before - after) / max(before, 1), after)
    return df


# ─── Feature / target split ───────────────────────────────────────────────────

def get_feature_columns(df: pd.DataFrame) -> list[str]:
    """Return all feature column names (everything except AQI target)."""
    return [c for c in df.columns if c != "AQI"]


def split_X_y(df: pd.DataFrame):
    """Return (X, y) arrays for model training."""
    feature_cols = get_feature_columns(df)
    X = df[feature_cols].values
    y = df["AQI"].values
    return X, y, feature_cols


def train_test_split_temporal(df: pd.DataFrame, test_days: int = 90):
    """
    Split into train/test preserving temporal order.
    The most recent `test_days` rows form the test set.
    Never shuffles — that would leak future data into training.
    """
    split_idx = len(df) - test_days
    train = df.iloc[:split_idx]
    test  = df.iloc[split_idx:]
    logger.info("Temporal split: %d train rows / %d test rows", len(train), len(test))
    return train, test


# ─── Standalone execution ─────────────────────────────────────────────────────

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-8s  %(message)s",
        datefmt="%H:%M:%S",
    )

    parser = argparse.ArgumentParser(description="AirCast: Preprocess Ahmedabad AQI data")
    parser.add_argument(
        "--input", type=Path,
        default=DATA_RAW_DIR / "city_day.csv",
        help="Path to raw Kaggle city_day.csv",
    )
    parser.add_argument(
        "--output", type=Path,
        default=PROCESSED_CSV,
        help="Destination for processed CSV",
    )
    args = parser.parse_args()

    df = build_features(raw_csv=args.input)
    out = save_processed(df, path=args.output)

    print(f"\n✓ Processed data saved to: {out}")
    print(f"  Rows: {len(df)}")
    print(f"  Date range: {df.index.min().date()} → {df.index.max().date()}")
    print(f"\n  Feature columns ({len(get_feature_columns(df))}):")
    for col in get_feature_columns(df):
        non_null = df[col].notna().sum()
        print(f"    {col:30s}  non-null: {non_null}")
    print(f"\n  Target (AQI):")
    print(f"    min={df['AQI'].min():.1f}  max={df['AQI'].max():.1f}"
          f"  mean={df['AQI'].mean():.1f}  std={df['AQI'].std():.1f}")
