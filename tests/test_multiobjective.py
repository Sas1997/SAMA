"""Tests for the multi-objective mode (NSGA-II via pymoo) and the scalar fitness.

The optimizer is first checked on problems whose Pareto front is known
analytically, then on the SAMA design problem itself.
"""
import re
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from samapy.core.Input_Data import InData
from samapy.optimizers.multiobjective import (MultiObjectiveOptimizer, _epsilon_nondominated,
                                              hull_gaps, select_from_front, supported_designs)

CONFIG = Path(__file__).resolve().parent / "data" / "example_config.yaml"


def schaffer(x):
    """Schaffer N.1: f1 = x^2, f2 = (x - 2)^2; Pareto-optimal for 0 <= x <= 2."""
    x = float(x[0])
    return {"NPC": x ** 2, "LEM": (x - 2.0) ** 2, "constraints": {}}


def schaffer_constrained(x):
    """Same, with the constraint x >= 1; Pareto-optimal for 1 <= x <= 2."""
    m = schaffer(x)
    m["constraints"] = {"x_min": 1.0 - float(x[0])}
    return m


def run(evaluate, objectives=("npc", "lem"), seed=1, **kw):
    return MultiObjectiveOptimizer(objectives=list(objectives), n_pop=40, n_gen=60, seed=seed,
                                   var_min=[-5.0], var_max=[5.0], evaluate=evaluate,
                                   generate_results=False, verbose=False, **kw).optimize()


def assert_mutually_nondominated(F_min):
    for i in range(len(F_min)):
        dominated = np.all(F_min <= F_min[i], axis=1) & np.any(F_min < F_min[i], axis=1)
        assert not dominated.any(), f"design {i} is dominated"


def test_recovers_known_pareto_front():
    r = run(schaffer, epsilon=0)
    x = r.front_X[:, 0]
    assert r.feasible
    assert len(x) >= 20
    assert np.all((x > -0.02) & (x < 2.02)), "all reported designs must lie on the Pareto set [0, 2]"
    assert x.min() < 0.1 and x.max() > 1.9, "the front must span the whole Pareto set"
    assert_mutually_nondominated(r.front_F)


def test_supported_designs_on_known_fronts():
    x = np.linspace(0.0, 1.0, 11)
    # convex and linear fronts: every point minimizes some weighted sum
    assert supported_designs(np.c_[x, (1 - x) ** 2]).all()
    assert supported_designs(np.c_[x, 1 - x]).all()
    # concave front f2 = 1 - f1^2: only the two end points are supported
    s = supported_designs(np.c_[x, 1 - x ** 2])
    assert s[0] and s[-1] and not s[1:-1].any()
    # three objectives: a point above the plane through the others is unsupported
    F = np.array([[0.0, 0.5, 0.5], [0.5, 0.0, 0.5], [0.5, 0.5, 0.0], [0.4, 0.4, 0.4]])
    assert list(supported_designs(F)) == [True, True, True, False]
    # the classification does not depend on the units of the objectives
    G = np.c_[x * 1e6, (1 - x ** 2) * 1e-3]
    assert list(supported_designs(G)) == list(s)
    # x = 0.5 on the concave front: with equal weights its weighted sum (0.625)
    # exceeds that of either end point (0.5) by 0.125, and no weights do better
    assert hull_gaps(np.c_[x, 1 - x ** 2])[5] == pytest.approx(0.125, abs=1e-9)
    # a design 0.5% of the range above a straight front is within the default
    # 1% tolerance of an approximate front; with tolerance 0 it is unsupported
    H = np.array([[0.0, 1.0], [0.5, 0.505], [1.0, 0.0]])
    assert supported_designs(H).all()
    assert list(supported_designs(H, tolerance=0)) == [True, False, True]


def concave(x):
    """f1 = x, f2 = 1 - x^2 on [0, 1]: every x is Pareto-optimal and the front is concave."""
    x = float(x[0])
    return {"NPC": x, "LEM": 1.0 - x ** 2, "constraints": {}}


def test_pareto_mode_finds_designs_no_weighted_sum_can_return():
    r = MultiObjectiveOptimizer(objectives=["npc", "lem"], n_pop=40, n_gen=60, seed=1,
                                var_min=[0.0], var_max=[1.0], evaluate=concave, epsilon=0,
                                generate_results=False, verbose=False).optimize()
    x = r.front_X[:, 0]
    assert len(x) >= 20 and x.min() < 0.05 and x.max() > 0.95
    assert_mutually_nondominated(r.front_F)
    interior = (x > 0.05) & (x < 0.95)
    assert interior.sum() >= 15
    # the interior of a concave front is unsupported
    assert not r.supported[interior].any()
    assert r.supported[np.argmin(x)] and r.supported[np.argmax(x)]
    # a weighted sum, minimized exactly over a fine grid, only ever returns an end point
    grid = np.linspace(0.0, 1.0, 2001)
    f = np.c_[grid, 1.0 - grid ** 2]
    for w in np.linspace(0.0, 1.0, 101):
        best = grid[np.argmin(w * f[:, 0] + (1 - w) * f[:, 1])]
        assert best in (0.0, 1.0)


def test_reported_front_keeps_its_ends_and_covers_every_design():
    # 21 designs along a front, each with 1 percentage point more renewable
    # fraction than the previous one (above the tolerance of 0.1 point).
    i = np.arange(21.0)
    tol = [5.0, 1e-3]
    # Each costs 6 more (above the cost tolerance of 5): all are trade-offs and kept.
    assert sorted(_epsilon_nondominated(np.c_[6 * i, -0.01 * i], 0.0, tol)) == list(range(21))
    # Each costs 1 more: a design may be left out only for a reported design that
    # costs at most 5 more and is no worse in renewable fraction. Comparing each
    # design with its neighbour instead would leave out all but the last one.
    F = np.c_[i, -0.01 * i]
    keep = _epsilon_nondominated(F, 0.0, tol)
    assert 0 in keep and 20 in keep and len(keep) >= 16
    for j in range(len(F)):
        assert any(np.all(F[k] <= F[j] + tol) for k in keep)
    # a design that adds a negligible renewable fraction at a large cost is not reported
    G = np.vstack([F, [1000.0, -0.2000001]])
    keep = _epsilon_nondominated(G, 0.0, tol)
    assert 21 not in keep and 20 in keep
    # each end of the front is reported to within the tolerance
    assert np.all(G[keep].min(axis=0) <= G.min(axis=0) + tol)
    # every design left out is within the tolerance of a reported one
    for j in range(len(G)):
        assert any(np.all(G[k] <= G[j] + tol) for k in keep)


def test_constraints_are_respected():
    r = run(schaffer_constrained, epsilon=0)
    x = r.front_X[:, 0]
    assert r.feasible and r.constraint_names == ["x_min"]
    assert np.all((x > 1.0 - 1e-6) & (x < 2.02))
    assert x.min() < 1.05 and x.max() > 1.9


def test_maximized_objective_is_reported_in_natural_units():
    # minimize cost x^2 and maximize the "renewable fraction" x on [0, 1]
    def evaluate(x):
        x = float(x[0])
        return {"NPC": x ** 2, "RE": x, "constraints": {}}

    r = MultiObjectiveOptimizer(objectives=["npc", "re"], n_pop=30, n_gen=40, seed=2,
                                var_min=[0.0], var_max=[1.0], evaluate=evaluate,
                                generate_results=False, verbose=False).optimize()
    npc, re_fraction = r.front_F[:, 0], r.front_F[:, 1]
    assert np.all(re_fraction >= 0) and re_fraction.max() > 0.95
    assert np.allclose(npc, re_fraction ** 2)
    # along the front, a higher renewable fraction costs more
    order = np.argsort(re_fraction)
    assert np.all(np.diff(npc[order]) >= 0)


def test_same_seed_gives_identical_front():
    a, b = run(schaffer, seed=5), run(schaffer, seed=5)
    np.testing.assert_array_equal(a.front_X, b.front_X)
    np.testing.assert_array_equal(a.front_F, b.front_F)


def test_selection_rules():
    # symmetric two-objective front (both minimized)
    F = np.array([[0.0, 4.0], [1.0, 1.0], [4.0, 0.0]])
    assert select_from_front(F, ["npc", "lem"], "min_npc") == 0
    assert select_from_front(F, ["npc", "lem"], "min_lem") == 2
    assert select_from_front(F, ["npc", "lem"], "compromise") == 1
    # a larger weight on an objective pulls the choice towards its best value
    assert select_from_front(F, ["npc", "lem"], [10.0, 1.0]) == 0
    with pytest.raises(ValueError):
        select_from_front(F, ["npc", "lem"], "min_re")
    with pytest.raises(ValueError):
        select_from_front(F, ["npc", "lem"], [1.0])


def test_invalid_objectives_are_rejected():
    with pytest.raises(ValueError):
        MultiObjectiveOptimizer(objectives=["npc"], generate_results=False)
    with pytest.raises(ValueError):
        MultiObjectiveOptimizer(objectives=["npc", "profit"], generate_results=False)


def test_fitness_equals_evaluate_and_weights_are_configurable(monkeypatch):
    from samapy.core.Fitness import evaluate, fitness

    X = np.array([12.0, 0.0, 2.0, 0.0, 8.0])
    m = evaluate(X)
    assert fitness(X) == m["fitness"]
    penalties = m["fitness"] - InData.NPC_weight * m["NPC"] - InData.EM_weight * InData.EM * m["LEM"]

    monkeypatch.setattr(InData, "NPC_weight", 2 * InData.NPC_weight)
    monkeypatch.setattr(InData, "EM", 1)
    m2 = evaluate(X)
    expected = InData.NPC_weight * m2["NPC"] + InData.EM_weight * m2["LEM"] + penalties
    assert m2["fitness"] == pytest.approx(expected, rel=1e-12)


def test_sama_design_problem_front_is_consistent():
    from samapy.core.Fitness import evaluate

    r = MultiObjectiveOptimizer(objectives=["npc", "re"], n_pop=20, n_gen=10, seed=3,
                                generate_results=False, verbose=False).optimize()
    assert len(r.front_X) >= 1
    assert set(r.constraint_names) >= {"lpsp", "re_min", "budget"}
    for x, f in zip(r.front_X, r.front_F):
        m = evaluate(x)
        assert f[0] == pytest.approx(m["NPC"]) and f[1] == pytest.approx(m["RE"])
        assert np.all(x >= r.front_X.min(axis=0))
    if r.feasible:
        assert all(v <= 1e-9 for m in r.front_metrics for v in m["constraints"].values())


@pytest.mark.slow
@pytest.mark.skipif(not CONFIG.is_file(), reason="example configuration not found")
def test_cli_multi_mode_writes_pareto_front(tmp_path):
    text = CONFIG.read_text(encoding="utf-8")
    text = re.sub(r"(?m)^MaxIt: \d+", "MaxIt: 10", text)
    text = re.sub(r"(?m)^nPop: \d+", "nPop: 12", text)
    (tmp_path / "samapy_config_COMPLETE.yaml").write_text(text, encoding="utf-8")
    result = subprocess.run([sys.executable, "-c", "from samapy.cli.runner import main; main()",
                             "--mode", "multi"], cwd=tmp_path, capture_output=True, text=True, encoding="utf-8", errors="replace",
                            timeout=1800)
    assert result.returncode == 0, result.stdout[-3000:] + result.stderr[-3000:]
    front = pd.read_csv(tmp_path / "samapy_outputs" / "pareto_front.csv")
    assert len(front) >= 1 and front["selected"].sum() == 1
    assert front["supported"].dtype == bool
    assert "Supported designs:" in result.stdout
    assert (tmp_path / "samapy_outputs" / "pareto_front.png").is_file()
    assert (tmp_path / "samapy_outputs" / "data" / "Outputforplotting.csv").is_file()
