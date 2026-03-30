"""
scheduler/daily_job.py
AirCast — Daily self-correction orchestrator for the Ahmedabad AQI prediction pipeline.

Execution order:
  1. Fetch yesterday's actual AQI from WAQI and upsert into DB
  2. Load yesterday's prediction from DB (made the previous day)
  3. Compare actual vs predicted — compute and log accuracy metrics
  4. Decide whether to retrain (MAE > threshold over last N days)
  5. If retraining → run full retrain, push new model to HF Hub
  6. Load best available model (new or existing)
  7. Fetch today's latest AQI reading as seed features
  8. Generate tomorrow's prediction and store it in DB

Run locally:
    python -m scheduler.daily_job

Run with forced retrain:
    python -m scheduler.daily_job --force-retrain

Run dry (no DB writes):
    python -m scheduler.daily_job --dry-run
"""

import argparse
import logging
import sys
from datetime import date, timedelta

import numpy as np

from config import (
    PRIMARY_STATION,
    RETRAIN_MAE_THRESHOLD,
    LAG_DAYS as FEATURE_LAGS,
    ROLLING_WINDOWS,
    validate_config,
    today_ist,
)
from pipeline.fetch_data import fetch_current_aqi
from pipeline.db import (
    upsert_actual,
    get_actuals,
    get_prediction_for_date,
    insert_prediction,
    log_performance,
    get_performance_history,
    get_latest_model_version,
    get_joined_chart_data,
)
from pipeline.evaluate import compute_metrics, should_retrain
from pipeline.model_store import load_model, push_model
from pipeline.train import retrain as run_retrain

# ─── Logging setup ────────────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)

EVAL_WINDOW_DAYS = 7   # number of recent days used to decide retraining

# validate_config() is called inside run() to avoid crashing at import time.


# ─── Step helpers ─────────────────────────────────────────────────────────────

def step_fetch_actual(yesterday: date, dry_run: bool) -> float | None:
    """Fetch the latest AQI from WAQI and store it in the DB.

    The reading is always persisted (via upsert) regardless of its date so
    that no data is silently lost.  Returns the AQI value only when the
    reading is actually for *yesterday* (needed for evaluation in step 3).
    """
    logger.info("━━ Step 1: Fetch actual AQI (target: %s) ━━", yesterday)

    data = fetch_current_aqi(PRIMARY_STATION)
    if data is None:
        logger.error("  Could not fetch AQI data from WAQI. Aborting step.")
        return None

    reading_date = data.get("date")
    actual_aqi = float(data.get("aqi") or data.get("value", 0))
    dominant = data.get("dominant_pollutant", "unknown")

    logger.info(
        "  WAQI reading: AQI %.1f on %s (dominant: %s)",
        actual_aqi, reading_date, dominant,
    )

    # Always persist the reading — upsert is safe for duplicate dates
    if not dry_run:
        upsert_actual({
            "date":               reading_date,
            "station":            PRIMARY_STATION,
            "aqi":                actual_aqi,
            "dominant_pollutant": dominant,
        })
        logger.info("  ✓ Actual for %s saved to DB", reading_date)
    else:
        logger.info("  [DRY-RUN] Skipped DB write")

    # For evaluation we need exactly yesterday's value
    if reading_date == yesterday:
        return actual_aqi

    logger.warning(
        "  Reading is from %s (wanted %s) — stored but eval will be skipped.",
        reading_date, yesterday,
    )
    return None


def _recent_prediction_bias(days: int = 21, min_points: int = 7) -> float:
    """
    Return mean residual (actual - predicted) over recent joined rows.

    Positive bias means model has been under-predicting; negative means over-predicting.
    """
    rows = get_joined_chart_data(days=days, station=PRIMARY_STATION)
    paired = [
        r for r in rows
        if r.get("predicted") is not None and r.get("actual_aqi") is not None
    ]
    if len(paired) < min_points:
        return 0.0

    residuals = [float(r["actual_aqi"]) - float(r["predicted"]) for r in paired]
    return float(np.mean(residuals))


def _calibrate_prediction(raw_pred: float, recent_actuals: list[float]) -> tuple[float, dict]:
    """
    Calibrate raw model output against recent observed AQI dynamics.

    This reduces large, systematic over/under-shooting while preserving trend signal.
    Returns (calibrated_prediction, diagnostics_dict).
    """
    pred = float(raw_pred)
    diag = {
        "center": None,
        "alpha": None,
        "clip_low": None,
        "clip_high": None,
    }

    if not recent_actuals:
        return float(np.clip(pred, 0.0, 500.0)), diag

    vals = np.array(recent_actuals[-21:], dtype=float)
    center = float(np.median(vals))
    spread = float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0

    # Dynamic shrinkage toward recent center.
    # More recent data -> trust model more. Large deviation -> shrink more.
    base_alpha = 0.35 + min(len(vals), 21) / 21.0 * 0.35  # 0.35 to 0.70
    deviation = abs(pred - center)
    if deviation > max(20.0, 1.5 * max(spread, 1.0)):
        alpha = base_alpha * 0.6
    else:
        alpha = base_alpha

    blended = center + alpha * (pred - center)

    # Keep prediction within a robust recent envelope.
    q10 = float(np.percentile(vals, 10))
    q90 = float(np.percentile(vals, 90))
    margin = max(15.0, 0.35 * max(spread, 1.0))
    low = q10 - margin
    high = q90 + margin

    calibrated = float(np.clip(blended, low, high))
    calibrated = float(np.clip(calibrated, 0.0, 500.0))

    diag.update({
        "center": round(center, 2),
        "alpha": round(alpha, 3),
        "clip_low": round(low, 2),
        "clip_high": round(high, 2),
    })
    return calibrated, diag


def step_load_prediction(yesterday: date) -> float | None:
    """Load the prediction we made yesterday for yesterday's date."""
    logger.info("━━ Step 2: Load yesterday's stored prediction (%s) ━━", yesterday)

    row = get_prediction_for_date(target_date=yesterday, station=PRIMARY_STATION)
    if row is None:
        logger.warning("  No stored prediction found for %s — skipping comparison", yesterday)
        return None

    predicted = float(row["predicted"])
    model_ver = row.get("model_ver", "unknown")
    logger.info("  Stored prediction: %.1f  (model: %s)", predicted, model_ver)
    return predicted


def step_evaluate(
    yesterday: date,
    actual: float,
    predicted: float,
    dry_run: bool,
) -> dict:
    """Compute point-in-time metrics and log them to DB."""
    logger.info("━━ Step 3: Evaluate accuracy ━━")

    m = compute_metrics(np.array([actual]), np.array([predicted]))
    mape_str = f"{m['mape']:.2f}%" if m["mape"] is not None else "N/A"
    logger.info(
        "  Point MAE: %.2f  RMSE: %.2f  MAPE: %s",
        m["mae"], m["rmse"], mape_str,
    )

    if not dry_run:
        log_performance(
            eval_date=yesterday,
            model_ver=get_latest_model_version() or "unknown",
            mae=m["mae"],
            rmse=m["rmse"],
            mape=m["mape"],
            retrain_triggered=False,
        )
        logger.info("  ✓ Performance logged to DB")
    else:
        logger.info("  [DRY-RUN] Skipped DB write")

    return m


def step_decide_retrain(force: bool) -> tuple[bool, str]:
    """
    Check rolling MAE over last EVAL_WINDOW_DAYS days.
    Returns (should_retrain_bool, reason_string).
    """
    logger.info("━━ Step 4: Decide retraining ━━")

    if force:
        reason = "forced via --force-retrain flag"
        logger.info("  Retraining FORCED: %s", reason)
        return True, reason

    history = get_performance_history(days=EVAL_WINDOW_DAYS)
    if not history:
        reason = "no performance history yet — skipping retrain"
        logger.info("  %s", reason)
        return False, reason

    recent_maes = [row["mae"] for row in history if row.get("mae") is not None]
    if not recent_maes:
        return False, "no MAE values in history"

    rolling_mae = float(np.mean(recent_maes))
    logger.info(
        "  Rolling MAE (%d-day): %.2f  |  threshold: %.2f",
        EVAL_WINDOW_DAYS, rolling_mae, RETRAIN_MAE_THRESHOLD,
    )

    if rolling_mae > RETRAIN_MAE_THRESHOLD:
        reason = f"rolling {EVAL_WINDOW_DAYS}d MAE {rolling_mae:.2f} > threshold {RETRAIN_MAE_THRESHOLD}"
        logger.info("  ✓ Retraining TRIGGERED: %s", reason)
        return True, reason
    else:
        reason = f"rolling MAE {rolling_mae:.2f} within threshold — no retrain needed"
        logger.info("  %s", reason)
        return False, reason


def step_retrain(reason: str, dry_run: bool) -> str | None:
    """Run full retraining + push to HF Hub. Returns new model version or None."""
    logger.info("━━ Step 5: Retraining model ━━")
    logger.info("  Reason: %s", reason)

    if dry_run:
        logger.info("  [DRY-RUN] Skipped actual retrain")
        return None

    try:
        model, feature_cols, metrics = run_retrain(push=True)  # trains + pushes to HF Hub
        new_ver = metrics.get("model_ver", f"retrain-{date.today().isoformat()}")
        logger.info("  ✓ Retrain complete — new model version: %s", new_ver)
        return new_ver
    except Exception as exc:
        logger.error("  ✗ Retrain failed: %s", exc, exc_info=True)
        return None


def step_update_performance_log(
    yesterday: date,
    old_ver: str,
    new_ver: str,
    reason: str,
    dry_run: bool,
) -> None:
    """Overwrite the performance row for yesterday to reflect the retrain event."""
    if dry_run or new_ver is None:
        return

    history = get_performance_history(days=2)
    # Use the last (most recent) row — history is ordered ascending by date
    row = history[-1] if history else {}

    log_performance(
        eval_date=yesterday,
        model_ver=old_ver,
        mae=row.get("mae", 0),
        rmse=row.get("rmse", 0),
        mape=row.get("mape", 0),
        retrain_triggered=True,
        retrain_reason=reason,
        new_model_ver=new_ver,
        promoted=True,
    )
    logger.info("  ✓ Performance row updated with retrain metadata")


def step_predict_tomorrow(
    today: date,
    dry_run: bool,
    preferred_model_ver: str | None = None,
) -> float | None:
    """
    Build a feature vector from today's latest AQI + recent actuals,
    run the model, and store the prediction for tomorrow.
    """
    logger.info("━━ Step 6: Predict tomorrow's AQI (%s) ━━", today + timedelta(days=1))

    # Load best available model
    result = load_model()
    if result is None:
        logger.error("  No model available — cannot generate prediction")
        return None

    model, feature_cols, model_metrics = result
    # BUG-FIX: metadata stores the version under "model_ver", not "version"
    model_ver = None
    if isinstance(model_metrics, dict):
        model_ver = model_metrics.get("model_ver") or model_metrics.get("version")
        if not model_ver and model_metrics.get("trained_on"):
            model_ver = f"local-{model_metrics.get('trained_on')}"
    if not model_ver:
        model_ver = preferred_model_ver or get_latest_model_version() or "unknown"

    # Pull last 30 days of actuals to build lag/rolling features
    actuals_rows = get_actuals(station=PRIMARY_STATION, days=35)

    if len(actuals_rows) < max(FEATURE_LAGS + [max(ROLLING_WINDOWS)]):
        logger.warning(
            "  Only %d days of actuals in DB — using live AQI reading for feature seed",
            len(actuals_rows),
        )
        live = fetch_current_aqi(PRIMARY_STATION)
        if live is None:
            logger.error("  Could not fetch live AQI — aborting prediction")
            return None
        current_aqi = float(live.get("aqi") or live.get("value", 100))
        recent_actual_values = [float(r["actual_aqi"]) for r in actuals_rows] + [current_aqi]
        # Minimal feature vector: repeat current AQI for all lag/rolling features
        feat_values = {col: current_aqi for col in feature_cols}
        _add_temporal_features(feat_values, today + timedelta(days=1))
    else:
        recent_actual_values = [float(r["actual_aqi"]) for r in actuals_rows]
        feat_values = _build_features_from_actuals(actuals_rows, today, feature_cols)

    # ── Safety check: detect train/inference feature mismatch ──────────────
    built_keys = set(feat_values.keys())
    model_keys = set(feature_cols)
    missing = model_keys - built_keys
    extra   = built_keys - model_keys
    if missing:
        logger.error("FEATURE MISMATCH: model expects %d features that inference didn't build: %s",
                     len(missing), sorted(missing))
        logger.error("These will default to 0.0 — predictions WILL be wrong!")
    if extra:
        logger.warning("Inference built %d extra features not used by model: %s",
                       len(extra), sorted(extra))

    # Convert to numpy row (model's feature_cols order)
    X = np.array([[feat_values.get(col, 0.0) for col in feature_cols]])
    raw_pred = float(model.predict(X)[0])
    bias = _recent_prediction_bias(days=21, min_points=7)
    # Clamp correction so transient spikes don't over-correct the forecast.
    bias = float(np.clip(bias, -80.0, 80.0))
    bias_adjusted = float(np.clip(raw_pred + bias, 0.0, 500.0))

    calibrated, cal_diag = _calibrate_prediction(bias_adjusted, recent_actual_values)
    predicted_aqi = round(calibrated, 1)

    logger.info(
        "  Predicted AQI for %s: %.1f  (raw: %.1f, bias_adj: %.1f, center: %s, alpha: %s, band: [%s, %s], model: %s)",
        today + timedelta(days=1),
        predicted_aqi,
        raw_pred,
        bias,
        cal_diag.get("center"),
        cal_diag.get("alpha"),
        cal_diag.get("clip_low"),
        cal_diag.get("clip_high"),
        model_ver,
    )

    if not dry_run:
        insert_prediction(
            target_date=today + timedelta(days=1),
            predicted_aqi=predicted_aqi,
            model_ver=model_ver,
            station=PRIMARY_STATION,
        )
        logger.info("  ✓ Prediction stored in DB")
    else:
        logger.info("  [DRY-RUN] Skipped DB write")

    return predicted_aqi


# ─── Feature building helpers ─────────────────────────────────────────────────

def _add_temporal_features(feat: dict, target_date: date) -> None:
    """Inject cyclical and seasonal features for a given date — must match preprocess.py exactly."""
    import math

    doy  = target_date.timetuple().tm_yday
    dow  = target_date.weekday()          # 0 = Monday
    m    = target_date.month
    # Week of year (ISO)
    woy  = target_date.isocalendar()[1]

    feat["dow_sin"]          = math.sin(2 * math.pi * dow  / 7)
    feat["dow_cos"]          = math.cos(2 * math.pi * dow  / 7)
    feat["month_sin"]        = math.sin(2 * math.pi * (m - 1) / 12)
    feat["month_cos"]        = math.cos(2 * math.pi * (m - 1) / 12)
    feat["woy_sin"]          = math.sin(2 * math.pi * (woy - 1) / 52)
    feat["woy_cos"]          = math.cos(2 * math.pi * (woy - 1) / 52)
    feat["day_of_year_norm"] = min((doy - 1) / 365.25, 1.0)
    feat["is_winter"]        = int(m in (11, 12, 1, 2))
    feat["is_summer"]        = int(m in (3, 4, 5))
    feat["is_monsoon"]       = int(m in (6, 7, 8, 9))
    feat["is_weekend"]       = int(dow >= 5)
    feat["is_festive_season"]= int(m in (10, 11))


def _build_features_from_actuals(
    actuals: list[dict],
    today: date,
    feature_cols: list[str],
) -> dict:
    """
    Reconstruct the feature vector that would be used to predict tomorrow
    using recent actual AQI values from the DB.

    MUST generate the exact same features as preprocess._engineer_features()
    so there is zero train/inference skew.
    """
    # Sort actuals descending by date
    actuals_sorted = sorted(actuals, key=lambda r: r["date"], reverse=True)
    aqi_series = [float(r["actual_aqi"]) for r in actuals_sorted]

    feat: dict = {}
    target_date = today + timedelta(days=1)

    # Current (today's) AQI — most recent reading
    current = aqi_series[0] if aqi_series else 100.0

    # ── Lag features ───────────────────────────────────────────────────────
    for lag in FEATURE_LAGS:
        feat[f"aqi_lag_{lag}d"] = aqi_series[lag - 1] if len(aqi_series) >= lag else current

    # ── Rolling statistics (mean / std / min / max) ────────────────────────
    for window in ROLLING_WINDOWS:
        window_vals = aqi_series[:window]
        if window_vals:
            v = np.array(window_vals, dtype=float)
            feat[f"aqi_roll_{window}d"] = float(np.mean(v))
            feat[f"aqi_rstd_{window}d"] = float(np.std(v, ddof=1)) if len(v) > 1 else 0.0
            feat[f"aqi_rmin_{window}d"] = float(np.min(v))
            feat[f"aqi_rmax_{window}d"] = float(np.max(v))
        else:
            feat[f"aqi_roll_{window}d"] = current
            feat[f"aqi_rstd_{window}d"] = 0.0
            feat[f"aqi_rmin_{window}d"] = current
            feat[f"aqi_rmax_{window}d"] = current

    # ── Volatility ratio (7-day range / mean) ──────────────────────────────
    v7 = np.array(aqi_series[:7], dtype=float) if len(aqi_series) >= 3 else np.array([current])
    v7_mean = float(np.mean(v7))
    feat["aqi_volatility_7d"] = float((np.max(v7) - np.min(v7)) / v7_mean) if v7_mean > 0 else 0.0

    # ── AQI momentum (day-over-day change) ─────────────────────────────────
    if len(aqi_series) >= 2:
        feat["aqi_diff_1d"] = aqi_series[0] - aqi_series[1]
        feat["aqi_pct_1d"]  = feat["aqi_diff_1d"] / aqi_series[1] if aqi_series[1] != 0 else 0.0
    else:
        feat["aqi_diff_1d"] = 0.0
        feat["aqi_pct_1d"]  = 0.0

    if len(aqi_series) >= 8:
        feat["aqi_diff_7d"] = aqi_series[0] - aqi_series[7]
    else:
        feat["aqi_diff_7d"] = 0.0

    # ── Exponential Moving Average ─────────────────────────────────────────
    if aqi_series:
        # Compute EMA manually over the reversed series (oldest first)
        rev = list(reversed(aqi_series))
        for span, label in [(7, "aqi_ema_7d"), (14, "aqi_ema_14d")]:
            alpha = 2 / (span + 1)
            ema = rev[0]
            for val in rev[1:]:
                ema = alpha * val + (1 - alpha) * ema
            feat[label] = ema
    else:
        feat["aqi_ema_7d"]  = current
        feat["aqi_ema_14d"] = current

    # ── Temporal / cyclical features ───────────────────────────────────────
    _add_temporal_features(feat, target_date)
    return feat


# ─── Main orchestrator ────────────────────────────────────────────────────────

def run(force_retrain: bool = False, dry_run: bool = False) -> None:
    validate_config()
    today     = today_ist()
    yesterday = today - timedelta(days=1)

    logger.info("╔══════════════════════════════════════════════╗")
    logger.info("║   AQI Daily Job — %s                ║", today.isoformat())
    logger.info("╚══════════════════════════════════════════════╝")

    # ── 1. Fetch actual AQI for yesterday ──────────────────────────────────────
    actual = step_fetch_actual(yesterday, dry_run)

    # ── 2. Load stored prediction for yesterday ────────────────────────────────
    predicted = step_load_prediction(yesterday)

    # ── 3. Evaluate (only if we have both actual and prediction) ───────────────
    if actual is not None and predicted is not None:
        step_evaluate(yesterday, actual, predicted, dry_run)
    else:
        logger.warning("Skipping evaluation — missing actual or prediction for %s", yesterday)

    # ── 4. Decide retraining ───────────────────────────────────────────────────
    do_retrain, reason = step_decide_retrain(force=force_retrain)

    # ── 5. Retrain if needed ───────────────────────────────────────────────────
    new_ver = None
    if do_retrain:
        old_ver = get_latest_model_version() or "unknown"
        new_ver = step_retrain(reason, dry_run)
        if new_ver:
            step_update_performance_log(yesterday, old_ver, new_ver, reason, dry_run)

    # ── 6. Predict tomorrow ────────────────────────────────────────────────────
    predicted_tomorrow = step_predict_tomorrow(
        today,
        dry_run,
        preferred_model_ver=new_ver,
    )

    # ── Summary ────────────────────────────────────────────────────────────────
    logger.info("")
    logger.info("╔══════════════════════════════════════════════╗")
    logger.info("║   DAILY JOB COMPLETE                        ║")
    logger.info("╠══════════════════════════════════════════════╣")
    logger.info("║   Yesterday actual   : %-20s  ║", f"{actual:.1f}" if actual else "N/A")
    logger.info("║   Yesterday predicted: %-20s  ║", f"{predicted:.1f}" if predicted else "N/A")
    logger.info("║   Retrain triggered  : %-20s  ║", str(do_retrain))
    if new_ver:
        logger.info("║   New model version  : %-20s  ║", new_ver)
    logger.info("║   Tomorrow forecast  : %-20s  ║", f"{predicted_tomorrow:.1f}" if predicted_tomorrow else "N/A")
    logger.info("╚══════════════════════════════════════════════╝")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="AQI daily self-correction job")
    parser.add_argument("--force-retrain", action="store_true",
                        help="Force retraining regardless of current metrics")
    parser.add_argument("--dry-run", action="store_true",
                        help="Run pipeline without writing anything to the DB")
    args = parser.parse_args()

    run(force_retrain=args.force_retrain, dry_run=args.dry_run)
