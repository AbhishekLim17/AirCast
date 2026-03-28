from datetime import date

import scheduler.daily_job as daily_job


def test_step_fetch_actual_skips_misaligned_fallback(monkeypatch):
    target = date(2026, 3, 27)

    monkeypatch.setattr(daily_job, "fetch_yesterday_aqi", lambda station: None)
    monkeypatch.setattr(
        daily_job,
        "fetch_current_aqi",
        lambda station: {
            "date": date(2026, 3, 28),
            "aqi": 120.0,
            "dominant_pollutant": "pm25",
        },
    )

    called = {"upsert": False}

    def _upsert(_):
        called["upsert"] = True

    monkeypatch.setattr(daily_job, "upsert_actual", _upsert)

    out = daily_job.step_fetch_actual(target, dry_run=False)

    assert out is None
    assert called["upsert"] is False


def test_recent_prediction_bias_uses_joined_rows(monkeypatch):
    monkeypatch.setattr(
        daily_job,
        "get_joined_chart_data",
        lambda days, station: [
            {"predicted": 200.0, "actual_aqi": 120.0},
            {"predicted": 210.0, "actual_aqi": 130.0},
            {"predicted": 220.0, "actual_aqi": 140.0},
            {"predicted": 230.0, "actual_aqi": 150.0},
            {"predicted": 240.0, "actual_aqi": 160.0},
            {"predicted": 250.0, "actual_aqi": 170.0},
            {"predicted": 260.0, "actual_aqi": 180.0},
        ],
    )

    bias = daily_job._recent_prediction_bias(days=21, min_points=7)

    # Mean(actual - predicted) across rows above = -80.0
    assert bias == -80.0


def test_calibrate_prediction_reduces_extreme_overprediction():
    recent = [72.0, 81.0, 95.0, 88.0, 76.0, 90.0, 84.0, 79.0, 86.0]
    raw = 260.0

    out, diag = daily_job._calibrate_prediction(raw, recent)

    assert out < raw
    assert 0.0 <= out <= 500.0
    assert diag["center"] is not None
    assert diag["alpha"] is not None


def test_calibrate_prediction_without_recent_data_is_passthrough():
    out, diag = daily_job._calibrate_prediction(123.4, [])

    assert out == 123.4
    assert diag["center"] is None
