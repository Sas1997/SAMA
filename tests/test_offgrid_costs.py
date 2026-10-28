"""Off-grid systems must not be charged for grid electricity.

Regression test: the heat pump heating and cooling grid costs in Fitness.py
were not multiplied by (Grid > 0), so an off-grid system with a heat pump was
charged for grid electricity it cannot buy. A per-kWh grid tax that only
applies to grid purchases must therefore have no effect on the objective of
an off-grid system.

Each case runs in a separate Python process because applying a configuration
changes the shared InData object.
"""
import re
import subprocess
import sys
from pathlib import Path

import pytest

CONFIG = Path(__file__).resolve().parents[1] / "samapy_config_COMPLETE.yaml"

SCRIPT = r"""
import io, contextlib, sys
import numpy as np
from samapy.cli.config_loader import load_config, apply_config
from samapy.cli import runner
with contextlib.redirect_stdout(io.StringIO()):
    d = apply_config(load_config(sys.argv[1]))
    runner._patch_fitness(d)
import samapy.core.Fitness as F
X = np.array([float(v) for v in sys.argv[2].split(',')])
base = F.fitness(X)
F.Grid_Tax_amount = F.Grid_Tax_amount * 100 + 1.0
taxed = F.fitness(X)
print(int(F.Grid), int(F.HP), repr(float(base)), repr(float(taxed)))
"""


# Designs [PV modules, wind turbines, batteries, generator kW, inverter kW]
# that satisfy the constraints in each case, so the objective is the cost
# itself rather than a constraint penalty.
OFFGRID_DESIGN = "30,0,40,5,10"
ONGRID_DESIGN = "12,0,2,0,8"


def _run(tmp_path, grid, design):
    text = CONFIG.read_text(encoding="utf-8")
    text = re.sub(r"(?m)^Grid: \d+", f"Grid: {grid}", text)
    text = re.sub(r"(?m)^NEM: \d+", f"NEM: {grid}", text)
    text = re.sub(r"(?m)^HP: \d+", "HP: 1", text)
    text = re.sub(r"(?m)^EV: \d+", "EV: 0", text)
    cfg = tmp_path / "config.yaml"
    cfg.write_text(text, encoding="utf-8")
    result = subprocess.run([sys.executable, "-c", SCRIPT, str(cfg), design], cwd=tmp_path,
                            capture_output=True, text=True, timeout=900)
    assert result.returncode == 0, result.stdout + result.stderr
    g, hp, base, taxed = result.stdout.strip().splitlines()[-1].split()
    assert (int(g), int(hp)) == (grid, 1)
    return float(base), float(taxed)


@pytest.mark.skipif(not CONFIG.is_file(), reason="example configuration not found")
def test_grid_tax_does_not_affect_offgrid_objective(tmp_path):
    base, taxed = _run(tmp_path, grid=0, design=OFFGRID_DESIGN)
    assert base < 100, "design should be feasible (objective not dominated by penalties)"
    assert taxed == base


@pytest.mark.skipif(not CONFIG.is_file(), reason="example configuration not found")
def test_grid_tax_still_affects_grid_connected_objective(tmp_path):
    # Sanity check that the test above is meaningful: with a grid, the tax counts.
    base, taxed = _run(tmp_path, grid=1, design=ONGRID_DESIGN)
    assert base < 100, "design should be feasible (objective not dominated by penalties)"
    assert taxed > base
