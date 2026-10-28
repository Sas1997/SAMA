"""Short SAMA optimizations run with the settings they are given.

Regression test for issue #12: settings changed on InData before an optimizer
is constructed must be the settings that actually run, even when the
optimizer module was imported earlier.
"""
import importlib
import os

import numpy as np
import pytest

from samapy.core.Input_Data import InData

# (module, class) for each optimizer. The modules are imported here, before
# InData is changed, which is exactly the situation that caused issue #12.
OPTIMIZERS = {
    "pso": ("samapy.optimizers.swarm", "Swarm"),
    "ade": ("samapy.optimizers.AdvancedDifferentialEvolution", "AdvancedDifferentialEvolution"),
    "abc": ("samapy.optimizers.ArtificialBeeColony", "ImprovedArtificialBeeColony"),
    "gwo": ("samapy.optimizers.GreyWolfOptimizer", "GreyWolfOptimizer"),
}
MODULES = {name: importlib.import_module(mod) for name, (mod, _) in OPTIMIZERS.items()}

QUICK = {"MaxIt": 10, "nPop": 20, "Run_Time": 1}
DEFAULTS = {key: getattr(InData, key) for key in QUICK}


@pytest.fixture
def quick_settings(monkeypatch):
    """Reduce the run size on InData; monkeypatch restores the originals afterwards."""
    for key, value in QUICK.items():
        monkeypatch.setattr(InData, key, value)
    return QUICK


@pytest.fixture
def counted_sama_cost(monkeypatch):
    """Count evaluations of the real SAMA objective without changing its result."""
    from samapy.core.Fitness import fitness

    calls = {"n": 0}

    def counting_fitness(x):
        calls["n"] += 1
        return fitness(x)

    for module in MODULES.values():
        monkeypatch.setattr(module, "cost_function", counting_fitness)
    return calls


@pytest.mark.parametrize("name", sorted(OPTIMIZERS))
def test_optimizer_uses_settings_set_before_construction(name, quick_settings, counted_sama_cost):
    cls = getattr(MODULES[name], OPTIMIZERS[name][1])
    opt = cls(seed=1, generate_results=False, verbose=False)

    # The optimizer reports the reduced settings ...
    assert (opt.MaxIt, opt.nPop, opt.Run_Time) == (10, 20, 1)

    X, cost = opt.optimize()

    # ... and actually ran them.
    assert len(opt.solution_cost_curve) == 1
    assert len(opt.solution_cost_curve[0]) == 10
    if name in ("pso", "gwo", "ade"):
        # initial population + one evaluation per member per iteration
        assert counted_sama_cost["n"] == 20 * (10 + 1)
    else:
        # ABC: employed and onlooker phases (20 per iteration) plus scouts
        assert 20 * 10 <= counted_sama_cost["n"] <= 20 * (10 + 1) + 10 * 10

    assert np.all(np.isfinite(X)) and np.isfinite(cost)
    assert np.all(X >= opt.VarMin - 1e-9) and np.all(X <= opt.VarMax + 1e-9)


def test_settings_are_restored_after_test():
    # Runs after the tests above; monkeypatch must have restored InData.
    assert {key: getattr(InData, key) for key in QUICK} == DEFAULTS


@pytest.mark.parametrize("name", ["ade", "abc"])
def test_population_too_small_is_rejected(name):
    cls = getattr(MODULES[name], OPTIMIZERS[name][1])
    with pytest.raises(ValueError):
        cls(n_pop=5, max_it=1, run_time=1, generate_results=False, verbose=False)


@pytest.mark.slow
def test_quick_workflow_writes_results(quick_settings, tmp_path, monkeypatch):
    """Full workflow on the default SAMA case: optimize, then write figures and data."""
    monkeypatch.chdir(tmp_path)
    from samapy.optimizers.swarm import Swarm

    opt = Swarm(seed=1, verbose=False)
    X, cost = opt.optimize()

    # the run matches the reduced settings it was given
    assert (opt.MaxIt, opt.nPop, opt.Run_Time) == (10, 20, 1)
    assert len(opt.solution_cost_curve) == 1 and len(opt.solution_cost_curve[0]) == 10
    assert np.isfinite(cost)
    assert os.path.isfile(tmp_path / "samapy_outputs" / "Optimization.png")
    assert os.path.isfile(tmp_path / "samapy_outputs" / "data" / "Outputforplotting.csv")
