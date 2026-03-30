"""
pipeline/db.py
All Supabase database interactions for the AQI system.
One function per logical operation — no raw queries scattered across files.

Usage:
    from pipeline.db import get_client, upsert_actual, insert_prediction, ...
"""

import logging
from datetime import date, timedelta
from typing import Optional

from supabase import create_client, Client

from config import SUPABASE_URL, SUPABASE_KEY, today_ist

logger = logging.getLogger(__name__)


# ─── Client ───────────────────────────────────────────────────────────────────────

_client: Client | None = None


def get_client() -> Client:
    """Return a Supabase client, creating one if needed.

    Unlike ``@lru_cache``, this allows reconnection when ``reset_client()``
    is called after a transient connection failure.
    """
    global _client
    if _client is not None:
        return _client
    if not SUPABASE_URL or not SUPABASE_KEY:
        raise EnvironmentError(
            "SUPABASE_URL and SUPABASE_KEY must be set before calling get_client(). "
            "Check your .env file or GitHub Actions Secrets."
        )
    _client = create_client(SUPABASE_URL, SUPABASE_KEY)
    return _client


def reset_client() -> None:
    """Discard the cached Supabase client so the next call creates a fresh one."""
    global _client
    _client = None


# ─── Actuals ──────────────────────────────────────────────────────────────────

def upsert_actual(reading: dict) -> None:
    """
    Insert or update a real AQI observation in the `actuals` table.

    `reading` is the dict returned by fetch_data.fetch_current_aqi().
    The unique constraint is (date, station) — duplicate calls are safe.
    """
    row = {
        "date":               reading["date"].isoformat(),
        "station":            reading["station"],
        "actual_aqi":         reading["aqi"],
        "dominant_pollutant": reading.get("dominant_pollutant"),
    }
    try:
        get_client().table("actuals").upsert(row, on_conflict="date,station").execute()
        logger.info("Upserted actual AQI %.1f for '%s' on %s", reading["aqi"], reading["station"], reading["date"])
    except Exception as exc:
        logger.error("Failed to upsert actual for station '%s': %s", reading["station"], exc)
        raise


def get_actuals(days: int = 30, station: str = "ahmedabad") -> list[dict]:
    """
    Return the last `days` actual AQI rows for a station, ordered by date ascending.
    """
    since = (today_ist() - timedelta(days=days)).isoformat()
    try:
        response = (
            get_client()
            .table("actuals")
            .select("date, actual_aqi, dominant_pollutant")
            .eq("station", station)
            .gte("date", since)
            .order("date", desc=False)
            .execute()
        )
        return response.data or []
    except Exception as exc:
        logger.error("Failed to fetch actuals: %s", exc)
        return []


def get_actual_for_date(target_date: date, station: str = "ahmedabad") -> Optional[float]:
    """Return the actual AQI for a specific date, or None if not found."""
    try:
        response = (
            get_client()
            .table("actuals")
            .select("actual_aqi")
            .eq("date", target_date.isoformat())
            .eq("station", station)
            .limit(1)
            .execute()
        )
        rows = response.data or []
        return float(rows[0]["actual_aqi"]) if rows else None
    except Exception as exc:
        logger.error("Failed to fetch actual for %s: %s", target_date, exc)
        return None


# ─── Predictions ──────────────────────────────────────────────────────────────

def insert_prediction(target_date: date, predicted_aqi: float, model_ver: str,
                      station: str = "ahmedabad") -> None:
    """
    Store the model's prediction for a future date.
    Uses upsert so re-running a job does not create duplicate rows.
    """
    row = {
        "date":      target_date.isoformat(),
        "station":   station,
        "predicted": round(predicted_aqi, 2),
        "model_ver": model_ver,
    }
    try:
        get_client().table("predictions").upsert(row, on_conflict="date,station").execute()
        logger.info("Stored prediction %.1f for '%s' on %s (model %s)", predicted_aqi, station, target_date, model_ver)
    except Exception as exc:
        logger.error("Failed to insert prediction: %s", exc)
        raise


def get_prediction_for_date(target_date: date, station: str = "ahmedabad") -> Optional[dict]:
    """Return the prediction row for a specific date, or None."""
    try:
        response = (
            get_client()
            .table("predictions")
            .select("predicted, model_ver")
            .eq("date", target_date.isoformat())
            .eq("station", station)
            .limit(1)
            .execute()
        )
        rows = response.data or []
        return rows[0] if rows else None
    except Exception as exc:
        logger.error("Failed to fetch prediction for %s: %s", target_date, exc)
        return None


def get_predictions(days: int = 30, station: str = "ahmedabad") -> list[dict]:
    """Return the last `days` prediction rows, ordered by date ascending."""
    since = (today_ist() - timedelta(days=days)).isoformat()
    try:
        response = (
            get_client()
            .table("predictions")
            .select("date, predicted, model_ver")
            .eq("station", station)
            .gte("date", since)
            .order("date", desc=False)
            .execute()
        )
        return response.data or []
    except Exception as exc:
        logger.error("Failed to fetch predictions: %s", exc)
        return []


# ─── Model Performance ────────────────────────────────────────────────────────

def log_performance(eval_date: date, model_ver: str, mae: float, rmse: float,
                    mape: float, retrain_triggered: bool = False,
                    retrain_reason: str = None, new_model_ver: str = None,
                    new_mae: float = None, promoted: bool = False) -> None:
    """
    Write one evaluation record to the `model_performance` table.
    Upsert on eval_date — re-running the daily job updates the existing row.
    """
    row = {
        "eval_date":           eval_date.isoformat(),
        "model_ver":           model_ver,
        "mae":                 round(mae, 4),
        "rmse":                round(rmse, 4),
        "mape":                round(mape, 4) if mape is not None else None,
        "retrain_triggered":   retrain_triggered,
        "retrain_reason":      retrain_reason,
        "new_model_ver":       new_model_ver,
        "new_mae":             round(new_mae, 4) if new_mae is not None else None,
        "promoted":            promoted,
    }
    try:
        get_client().table("model_performance").upsert(row, on_conflict="eval_date").execute()
        logger.info(
            "Logged performance for %s — MAE=%.2f  retrain=%s  promoted=%s",
            eval_date, mae, retrain_triggered, promoted,
        )
    except Exception as exc:
        logger.error("Failed to log performance for %s: %s", eval_date, exc)
        raise


def get_performance_history(days: int = 30) -> list[dict]:
    """Return the last `days` model performance rows, oldest first."""
    since = (today_ist() - timedelta(days=days)).isoformat()
    try:
        response = (
            get_client()
            .table("model_performance")
            .select("eval_date, model_ver, mae, rmse, mape, retrain_triggered, retrain_reason, new_model_ver, new_mae, promoted")
            .gte("eval_date", since)
            .order("eval_date", desc=False)
            .execute()
        )
        return response.data or []
    except Exception as exc:
        logger.error("Failed to fetch performance history: %s", exc)
        return []


def get_latest_model_version() -> Optional[str]:
    """Return the model_ver from the most recent prediction row."""
    try:
        response = (
            get_client()
            .table("predictions")
            .select("model_ver")
            .order("created_at", desc=True)
            .limit(1)
            .execute()
        )
        rows = response.data or []
        return rows[0]["model_ver"] if rows else None
    except Exception as exc:
        logger.error("Failed to fetch latest model version: %s", exc)
        return None


def get_joined_chart_data(days: int = 30, station: str = "ahmedabad") -> list[dict]:
    """
    Return a merged list of {date, predicted, actual_aqi} for the dashboard chart.
    Uses a LEFT JOIN on predictions — every prediction date is included, with
    actual_aqi set to None when the actual hasn't been recorded yet.
    This prevents data gaps from silently disappearing off the chart.
    """
    predictions = {r["date"]: r["predicted"] for r in get_predictions(days, station)}
    actuals     = {r["date"]: r["actual_aqi"] for r in get_actuals(days, station)}

    merged = []
    for d in sorted(predictions):  # All prediction dates, not just the intersection
        merged.append({
            "date":       d,
            "predicted":  predictions[d],
            "actual_aqi": actuals.get(d),  # None when actual not yet available
        })
    return merged
