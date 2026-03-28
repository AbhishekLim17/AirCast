from config import get_aqi_category


def test_get_aqi_category_handles_float_boundaries():
    assert get_aqi_category(50.0)["label"] == "Good"
    assert get_aqi_category(50.5)["label"] == "Satisfactory"
    assert get_aqi_category(100.0)["label"] == "Satisfactory"
    assert get_aqi_category(100.1)["label"] == "Moderate"


def test_get_aqi_category_clamps_out_of_range_values():
    assert get_aqi_category(-10)["label"] == "Good"
    assert get_aqi_category(999)["label"] == "Severe"
