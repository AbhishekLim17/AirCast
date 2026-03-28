import numpy as np

from pipeline.evaluate import compute_metrics


def test_compute_metrics_basic_values():
    y_true = np.array([100, 200, 300], dtype=float)
    y_pred = np.array([110, 190, 330], dtype=float)

    metrics = compute_metrics(y_true, y_pred)

    assert metrics["mae"] == 16.666666666666668
    assert metrics["rmse"] == 19.148542155126762
    assert round(metrics["mape"], 6) == round(8.333333333333332, 6)


def test_compute_metrics_mape_none_when_all_actuals_zero():
    y_true = np.array([0.0, 0.0, 0.0])
    y_pred = np.array([1.0, 2.0, 3.0])

    metrics = compute_metrics(y_true, y_pred)

    assert metrics["mae"] == 2.0
    assert metrics["rmse"] == np.sqrt((1.0**2 + 2.0**2 + 3.0**2) / 3.0)
    assert metrics["mape"] is None


def test_compute_metrics_raises_on_length_mismatch():
    y_true = np.array([1.0, 2.0])
    y_pred = np.array([1.0])

    try:
        compute_metrics(y_true, y_pred)
        assert False, "Expected ValueError for length mismatch"
    except ValueError as exc:
        assert "Length mismatch" in str(exc)
