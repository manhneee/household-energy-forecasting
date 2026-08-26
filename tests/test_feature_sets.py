from src.data.feature_sets import resolve_columns


def test_none_excludes_weather():
    hist, fut = resolve_columns(
        "none",
        ["kwh_input", "hour_sin", "hour_cos", "temperature_2m", "shortwave_radiation"],
    )
    assert "kwh_input" in hist
    assert "temperature_2m" not in hist
    assert "kwh_input" not in fut


def test_selected_keeps_only_available_weather():
    hist, fut = resolve_columns("selected", ["kwh_input", "temperature_2m"])
    assert hist == ["kwh_input", "temperature_2m"]
    assert fut == ["temperature_2m"]
