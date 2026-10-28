"""Result generation writes the expected data and figure files."""
import os

import pandas as pd
import pytest

HOURS_PER_YEAR = 8760


@pytest.mark.slow
def test_gen_results_writes_data_and_figures(tmp_path):
    from samapy.results.Results import Gen_Results

    out = tmp_path / "results"
    # [PV modules, wind turbines, batteries, generator, inverter kW]
    Gen_Results([10, 0, 50, 1, 25], output_dir=str(out))

    csv_path = out / "data" / "Outputforplotting.csv"
    assert os.path.isfile(csv_path)
    assert len(pd.read_csv(csv_path)) == HOURS_PER_YEAR

    figs = list((out / "figs").iterdir())
    assert any(f.name == "Energy Distribution.png" for f in figs)
    assert all(f.stat().st_size > 0 for f in figs)
