from datetime import date

import scheduler.daily_job as daily_job


def test_step_fetch_actual_stores_latest_but_skips_eval_if_not_yesterday(monkeypatch):
    target = date(2026, 3, 27)

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
    assert called["upsert"] is True


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


def test_backfill_missing_performance_inserts_only_missing_dates(monkeypatch):
    monkeypatch.setattr(
        daily_job,
        "get_predictions",
        lambda days, station: [
            {"date": "2026-04-01", "predicted": 120.0, "model_ver": "m1"},
            {"date": "2026-04-02", "predicted": 110.0, "model_ver": "m1"},
        ],
    )
    monkeypatch.setattr(
        daily_job,
        "get_actuals",
        lambda days, station: [
            {"date": "2026-04-01", "actual_aqi": 100.0},
            {"date": "2026-04-02", "actual_aqi": 90.0},
        ],
    )
    monkeypatch.setattr(
        daily_job,
        "get_performance_history",
        lambda days: [{"eval_date": "2026-04-01"}],
    )

    calls = []

    def _log_performance(**kwargs):
        calls.append(kwargs)

    monkeypatch.setattr(daily_job, "log_performance", _log_performance)

    inserted = daily_job.step_backfill_missing_performance(days=30, dry_run=False)

    assert inserted == 1
    assert len(calls) == 1
    assert calls[0]["model_ver"] == "m1"


def test_adaptive_model_weight_favors_persistence_when_model_worse(monkeypatch):
    monkeypatch.setattr(
        daily_job,
        "get_joined_chart_data",
        lambda days, station: [
            {"date": "2026-04-01", "predicted": 200.0, "actual_aqi": 100.0},
            {"date": "2026-04-02", "predicted": 200.0, "actual_aqi": 110.0},
            {"date": "2026-04-03", "predicted": 200.0, "actual_aqi": 120.0},
            {"date": "2026-04-04", "predicted": 200.0, "actual_aqi": 130.0},
        ],
    )
    monkeypatch.setattr(
        daily_job,
        "get_actuals",
        lambda days, station: [
            {"date": "2026-03-31", "actual_aqi": 99.0},
            {"date": "2026-04-01", "actual_aqi": 100.0},
            {"date": "2026-04-02", "actual_aqi": 110.0},
            {"date": "2026-04-03", "actual_aqi": 120.0},
            {"date": "2026-04-04", "actual_aqi": 130.0},
        ],
    )

    w, diag = daily_job._adaptive_model_weight(days=21, min_points=4)

    assert w == 0.2
    assert diag["reason"] == "favor_persistence"


def test_adaptive_model_weight_favors_model_when_model_better(monkeypatch):
    monkeypatch.setattr(
        daily_job,
        "get_joined_chart_data",
        lambda days, station: [
            {"date": "2026-04-01", "predicted": 100.0, "actual_aqi": 101.0},
            {"date": "2026-04-02", "predicted": 110.0, "actual_aqi": 111.0},
            {"date": "2026-04-03", "predicted": 120.0, "actual_aqi": 121.0},
            {"date": "2026-04-04", "predicted": 130.0, "actual_aqi": 131.0},
        ],
    )
    monkeypatch.setattr(
        daily_job,
        "get_actuals",
        lambda days, station: [
            {"date": "2026-03-31", "actual_aqi": 70.0},
            {"date": "2026-04-01", "actual_aqi": 100.0},
            {"date": "2026-04-02", "actual_aqi": 110.0},
            {"date": "2026-04-03", "actual_aqi": 120.0},
            {"date": "2026-04-04", "actual_aqi": 130.0},
        ],
    )

    w, diag = daily_job._adaptive_model_weight(days=21, min_points=4)

    assert w == 0.8
    assert diag["reason"] == "favor_model"
