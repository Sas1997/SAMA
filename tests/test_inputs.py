"""Default input data loads with the expected hourly resolution and sensible values."""
import numpy as np

HOURS_PER_YEAR = 8760


def test_hourly_series_have_one_value_per_hour():
    from samapy.core.Input_Data import InData

    for name in ("Eload", "G", "T", "Vw"):
        series = np.asarray(getattr(InData, name))
        assert series.shape == (HOURS_PER_YEAR,), f"{name} has shape {series.shape}"
        assert np.all(np.isfinite(series)), f"{name} contains non-finite values"


def test_physical_ranges():
    from samapy.core.Input_Data import InData

    assert np.all(np.asarray(InData.Eload) >= 0), "electrical load must be non-negative"
    assert np.all(np.asarray(InData.G) >= 0), "irradiance must be non-negative"
    assert np.all(np.asarray(InData.Vw) >= 0), "wind speed must be non-negative"


def test_optimizer_settings_are_valid():
    from samapy.core.Input_Data import InData

    assert InData.nPop >= 6
    assert InData.MaxIt >= 1
    assert InData.Run_Time >= 1
    assert len(InData.VarMin) == len(InData.VarMax) == 5
    assert np.all(np.asarray(InData.VarMax) >= np.asarray(InData.VarMin))
