"""
pipeline/train.py
Train XGBoost for AirCast — Ahmedabad AQI dataset.

Steps:
  1. Load processed CSV
  2. Walk-forward time-series cross-validation to get honest baseline metrics
  3. Optuna hyperparameter search (CV-based objective)
  4. Train final model on full training window with best params
  5. Evaluate on held-out test set
  6. Save locally (and optionally push to HF Hub)

Usage:
    python -m pipeline.train                     # train + save locally
    python -m pipeline.train --push              # train + push to HF Hub
    python -m pipeline.train --trials 20         # fewer Optuna trials (faster)
"""

import argparse
import logging
import warnings
from datetime import date

import numpy as np
import xgboost as xgb
import optuna

from config import OPTUNA_TRIALS, CV_FOLDS, RETRAIN_WINDOW_DAYS, RETRAIN_TEST_DAYS
from pipeline.preprocess import (
    load_processed, split_X_y, train_test_split_temporal, get_feature_columns,
)
from pipeline.evaluate import compute_metrics
from pipeline.model_store import save_local, push_model

warnings.filterwarnings("ignore", category=UserWarning, module="xgboost")
optuna.logging.set_verbosity(optuna.logging.WARNING)

logger = logging.getLogger(__name__)


# ─── Walk-forward cross-validation ────────────────────────────────────────────

def walk_forward_cv(X: np.ndarray, y: np.ndarray,
                    n_folds: int, params: dict) -> dict:
    """
    Time-series walk-forward CV — never shuffles, never leaks future data.

    Each fold trains on everything before the validation window and evaluates
    on the next chunk. Uses early stopping to prevent overfitting even with
    large n_estimators. Returns averaged MAE/RMSE/MAPE across all folds.
    """
    fold_size = len(X) // (n_folds + 1)
    if fold_size < 2:
        raise ValueError(
            f"Not enough data for {n_folds}-fold CV: {len(X)} samples, "
            f"need at least {2 * (n_folds + 1)}"
        )
    maes, rmses, mapes = [], [], []

    # Extract early stopping rounds (not a model param)
    early_stop = params.pop("early_stopping_rounds", 50)
    n_est = params.get("n_estimators", 500)

    for fold in range(n_folds):
        train_end = fold_size * (fold + 1)
        val_end   = train_end + fold_size

        X_tr, y_tr = X[:train_end], y[:train_end]
        X_val, y_val = X[train_end:val_end], y[train_end:val_end]

        model = xgb.XGBRegressor(**params, verbosity=0, random_state=42)
        model.fit(
            X_tr, y_tr,
            eval_set=[(X_val, y_val)],
            verbose=False,
        )
        preds = model.predict(X_val)

        m = compute_metrics(y_val, preds)
        maes.append(m["mae"])
        rmses.append(m["rmse"])
        if m["mape"] is not None:
            mapes.append(m["mape"])

        logger.debug("  Fold %d/%d — MAE=%.2f  RMSE=%.2f",
                     fold + 1, n_folds, m["mae"], m["rmse"])

    # Re-insert early stopping so the dict stays intact for the caller
    params["early_stopping_rounds"] = early_stop

    return {
        "mae":  float(np.mean(maes)),
        "rmse": float(np.mean(rmses)),
        "mape": float(np.mean(mapes)) if mapes else None,
    }


# ─── Optuna objective ─────────────────────────────────────────────────────────

def _make_objective(X_train: np.ndarray, y_train: np.ndarray, n_folds: int):
    def objective(trial: optuna.Trial) -> float:
        params = {
            # ── Tree structure ──
            "n_estimators":           trial.suggest_int("n_estimators", 200, 2000),
            "max_depth":              trial.suggest_int("max_depth", 3, 12),
            "max_leaves":             trial.suggest_int("max_leaves", 0, 256),  # 0 = unlimited
            "grow_policy":            trial.suggest_categorical("grow_policy", ["depthwise", "lossguide"]),
            # ── Learning ──
            "learning_rate":          trial.suggest_float("learning_rate", 0.005, 0.3, log=True),
            "gamma":                  trial.suggest_float("gamma", 1e-5, 5.0, log=True),
            "min_child_weight":       trial.suggest_int("min_child_weight", 1, 15),
            # ── Sampling (anti-overfitting) ──
            "subsample":              trial.suggest_float("subsample", 0.5, 1.0),
            "colsample_bytree":       trial.suggest_float("colsample_bytree", 0.4, 1.0),
            "colsample_bylevel":      trial.suggest_float("colsample_bylevel", 0.4, 1.0),
            # ── Regularisation ──
            "reg_alpha":              trial.suggest_float("reg_alpha", 1e-5, 20.0, log=True),
            "reg_lambda":             trial.suggest_float("reg_lambda", 1e-5, 20.0, log=True),
            # ── Early stopping ──
            "early_stopping_rounds":  trial.suggest_int("early_stopping_rounds", 20, 80),
            # ── Fixed ──
            "tree_method":            "hist",
            "objective":              "reg:squarederror",
        }
        cv_results = walk_forward_cv(X_train, y_train, n_folds=n_folds, params=params)
        return cv_results["mae"]   # minimise MAE

    return objective


# ─── Main training function ───────────────────────────────────────────────────

def train(n_trials: int = OPTUNA_TRIALS,
          n_folds:  int = CV_FOLDS,
          push:     bool = False,
          window_days: int = None) -> tuple:
    """
    Full training pipeline.

    Args:
        n_trials:    Number of Optuna hyperparameter search trials.
        n_folds:     Walk-forward CV folds.
        push:        If True, push model to Hugging Face Hub after training.
        window_days: If set, train only on the last N days (retraining mode).

    Returns:
        (model, feature_cols, metrics_on_test)
    """
    logger.info("═══ Phase 4: Model Training ═══")

    # ── Load data ──────────────────────────────────────────────────────────
    df = load_processed()
    if df.empty:
        raise RuntimeError("No processed data found. Run pipeline.preprocess first.")

    if window_days:
        df = df.iloc[-window_days:]
        logger.info("Retraining on last %d rows", len(df))

    # ── Temporal train/test split ──────────────────────────────────────────
    test_days = min(RETRAIN_TEST_DAYS, len(df) // 3)  # never use more than 1/3 for test
    if test_days < 7:
        raise RuntimeError(
            f"Not enough data to split: {len(df)} rows, need at least 21. "
            "Increase RETRAIN_WINDOW_DAYS or gather more data."
        )
    train_df, test_df = train_test_split_temporal(df, test_days=test_days)
    X_train, y_train, feature_cols = split_X_y(train_df)
    X_test,  y_test,  _            = split_X_y(test_df)

    logger.info("Training set: %d rows | Test set: %d rows | Features: %d",
                len(X_train), len(X_test), len(feature_cols))

    # ── Optuna hyperparameter search ───────────────────────────────────────
    logger.info("Starting Optuna search — %d trials, %d CV folds …", n_trials, n_folds)
    study = optuna.create_study(direction="minimize",
                                sampler=optuna.samplers.TPESampler(seed=42))
    study.optimize(_make_objective(X_train, y_train, n_folds),
                   n_trials=n_trials, show_progress_bar=False)

    best_params = study.best_params
    best_cv_mae = study.best_value
    best_params["tree_method"] = "hist"

    logger.info("Best CV MAE: %.2f", best_cv_mae)
    logger.info("Best params: %s", best_params)

    # ── Train final model on full training window ──────────────────────────
    logger.info("Training final model on full training window …")
    # early_stopping_rounds is not an XGBRegressor constructor param — extract it
    final_params = {**best_params}
    final_params["tree_method"] = "hist"
    es_rounds = final_params.pop("early_stopping_rounds", 50)

    final_model = xgb.XGBRegressor(**final_params, verbosity=0, random_state=42)
    final_model.fit(
        X_train, y_train,
        eval_set=[(X_test, y_test)],
        verbose=False,
    )

    # ── Evaluate on held-out test set ─────────────────────────────────────
    test_preds = final_model.predict(X_test)
    test_metrics = compute_metrics(y_test, test_preds)

    mape_str = f"{test_metrics['mape']:.2f}%" if test_metrics["mape"] is not None else "N/A"
    logger.info(
        "Test set — MAE: %.2f  RMSE: %.2f  MAPE: %s",
        test_metrics["mae"], test_metrics["rmse"], mape_str,
    )

    # ── Feature importance (top 10) ───────────────────────────────────────
    importances = dict(zip(feature_cols,
                           final_model.feature_importances_.tolist()))
    top10 = sorted(importances.items(), key=lambda x: x[1], reverse=True)[:10]
    logger.info("Top 10 features by importance:")
    for feat, imp in top10:
        logger.info("  %-25s  %.4f", feat, imp)

    metrics = {
        **test_metrics,
        "cv_mae":         best_cv_mae,
        "best_params":    best_params,
        "feature_cols":   feature_cols,
        "top_features":   top10,
        "trained_on":     date.today().isoformat(),
        "train_rows":     len(X_train),
        "test_rows":      len(X_test),
    }

    # ── Save ───────────────────────────────────────────────────────────────
    save_local(final_model, feature_cols, metrics)

    if push:
        sha = push_model(final_model, feature_cols, metrics)
        metrics["model_ver"] = sha[:8]
        logger.info("Model pushed — version: %s", metrics["model_ver"])
    else:
        metrics["model_ver"] = f"local-{date.today().isoformat()}"

    return final_model, feature_cols, metrics


# ─── Retrain wrapper (called by daily_job.py) ─────────────────────────────────

def retrain(push: bool = True) -> tuple:
    """
    Retrain on the last RETRAIN_WINDOW_DAYS days of data.
    Returns (model, feature_cols, metrics).
    """
    logger.info("Retraining triggered — window: %d days", RETRAIN_WINDOW_DAYS)
    return train(
        n_trials=max(10, OPTUNA_TRIALS // 3),  # faster for daily retrains
        n_folds=3,
        push=push,
        window_days=RETRAIN_WINDOW_DAYS,
    )


# ─── Standalone execution ─────────────────────────────────────────────────────

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-8s  %(message)s",
        datefmt="%H:%M:%S",
    )

    parser = argparse.ArgumentParser(description="AirCast: Train Ahmedabad AQI XGBoost model")
    parser.add_argument("--trials", type=int, default=OPTUNA_TRIALS,
                        help="Number of Optuna trials (default: %(default)s)")
    parser.add_argument("--push", action="store_true",
                        help="Push trained model to Hugging Face Hub")
    args = parser.parse_args()

    model, feature_cols, metrics = train(n_trials=args.trials, push=args.push)

    print("\n" + "═" * 55)
    print("  TRAINING COMPLETE")
    print("═" * 55)
    print(f"  Test MAE  : {metrics['mae']:.2f}")
    print(f"  Test RMSE : {metrics['rmse']:.2f}")
    print(f"  Test MAPE : {metrics['mape']:.2f}%")
    print(f"  CV MAE    : {metrics['cv_mae']:.2f}")
    print(f"  Model ver : {metrics['model_ver']}")
    print(f"  Saved     : models/xgb_model.pkl")
    print("═" * 55)
