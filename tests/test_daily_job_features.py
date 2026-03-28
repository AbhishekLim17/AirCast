from datetime import date, timedelta

from config import LAG_DAYS, ROLLING_WINDOWS
from scheduler.daily_job import _build_features_from_actuals


def _expected_feature_columns():
    cols = []
    for lag in LAG_DAYS:
        cols.append(f"aqi_lag_{lag}d")
    for window in ROLLING_WINDOWS:
        cols.extend(
            [
                f"aqi_roll_{window}d",
                f"aqi_rstd_{window}d",
                f"aqi_rmin_{window}d",
                f"aqi_rmax_{window}d",
            ]
        )
    cols.extend(
        [
            "aqi_volatility_7d",
            "aqi_diff_1d",
            "aqi_pct_1d",
            "aqi_diff_7d",
            "aqi_ema_7d",
            "aqi_ema_14d",
            "dow_sin",
            "dow_cos",
            "month_sin",
            "month_cos",
            "woy_sin",
            "woy_cos",
            "day_of_year_norm",
            "is_winter",
            "is_summer",
            "is_monsoon",
            "is_weekend",
            "is_festive_season",
        ]
    )
    return cols


def test_build_features_covers_expected_columns():
    today = date(2026, 3, 28)
    actuals = []
    for i in range(35):
        actuals.append(
            {
                "date": (today - timedelta(days=i)).isoformat(),
                "actual_aqi": float(120 + i),
            }
        )

    feature_cols = _expected_feature_columns()
    feat = _build_features_from_actuals(actuals=actuals, today=today, feature_cols=feature_cols)

    missing = [c for c in feature_cols if c not in feat]
    assert missing == []
