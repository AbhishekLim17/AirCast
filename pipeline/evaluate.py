"""
pipeline/evaluate.py
Computes regression metrics for AQI predictions and writes results to the DB.
"""

import logging
from datetime import date

import numpy as np

logger = logging.getLogger(__name__)


# ─── Core metrics ─────────────────────────────────────────────────────────────

def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """
    Compute MAE, RMSE, and MAPE.

    Args:
        y_true: Array of actual AQI values.
        y_pred: Array of predicted AQI values (same length).

    Returns:
        {"mae": float, "rmse": float, "mape": float}
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    if len(y_true) == 0 or len(y_pred) == 0:
        raise ValueError("y_true and y_pred must not be empty")
    if len(y_true) != len(y_pred):
        raise ValueError(f"Length mismatch: y_true={len(y_true)}, y_pred={len(y_pred)}")

    mae  = float(np.mean(np.abs(y_true - y_pred)))
    rmse = float(np.sqrt(np.mean((y_true - y_pred) ** 2)))

    # MAPE: avoid division by zero — skip rows where actual AQI is 0
    nonzero = y_true != 0
    if nonzero.sum() == 0:
        mape = float("nan")
    else:
        mape = float(np.mean(np.abs((y_true[nonzero] - y_pred[nonzero]) / y_true[nonzero])) * 100)

    logger.info("Metrics — MAE: %.2f  RMSE: %.2f  MAPE: %.2f%%", mae, rmse, mape)
    return {"mae": mae, "rmse": rmse, "mape": mape}


def evaluate_and_log(y_true: np.ndarray, y_pred: np.ndarray,
                     model_ver: str, eval_date: date = None,
                     retrain_triggered: bool = False,
                     retrain_reason: str = None,
                     new_model_ver: str = None,
                     new_mae: float = None,
                     promoted: bool = False) -> dict:
    """
    Compute metrics and immediately persist them to the model_performance table.

    Returns the metrics dict.
    """
    from pipeline.db import log_performance

    metrics = compute_metrics(y_true, y_pred)
    eval_date = eval_date or date.today()

    log_performance(
        eval_date=eval_date,
        model_ver=model_ver,
        mae=metrics["mae"],
        rmse=metrics["rmse"],
        mape=metrics["mape"],
        retrain_triggered=retrain_triggered,
        retrain_reason=retrain_reason,
        new_model_ver=new_model_ver,
        new_mae=new_mae,
        promoted=promoted,
    )
    return metrics


def should_retrain(mae: float) -> tuple[bool, str]:
    """
    Decide whether the daily MAE warrants retraining.

    Returns (trigger: bool, reason: str)
    """
    from config import RETRAIN_MAE_THRESHOLD
    if mae > RETRAIN_MAE_THRESHOLD:
        reason = f"MAE {mae:.2f} exceeded threshold {RETRAIN_MAE_THRESHOLD}"
        logger.warning("Retrain triggered: %s", reason)
        return True, reason
    logger.info("No retrain needed — MAE %.2f <= threshold %.2f", mae, RETRAIN_MAE_THRESHOLD)
    return False, ""
