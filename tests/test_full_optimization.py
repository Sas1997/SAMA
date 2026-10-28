"""Full optimization of the default SAMA case with the default settings.

This runs the complete workflow (several minutes), so it is marked slow and
is not part of the default test run. Run it with:  pytest -m slow
"""
import os

import numpy as np
import pytest


@pytest.mark.slow
def test_full_default_optimization(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    from samapy.optimizers.swarm import Swarm

    opt = Swarm(seed=0, verbose=False)
    X, cost = opt.optimize()

    assert np.all(np.isfinite(X)) and np.isfinite(cost)
    assert np.all(X >= opt.VarMin - 1e-9) and np.all(X <= opt.VarMax + 1e-9)
    # the best cost never gets worse from one iteration to the next
    curve = np.asarray(opt.solution_cost_curve[0])
    assert np.all(np.diff(curve) <= 1e-12)
    assert os.path.isfile(tmp_path / "samapy_outputs" / "Optimization.png")
    assert os.path.isfile(tmp_path / "samapy_outputs" / "data" / "Outputforplotting.csv")
