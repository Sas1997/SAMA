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
from samapy.optimizers.multiobjective import MultiObjectiveOptimizer, select_from_front

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
    assert (tmp_path / "samapy_outputs" / "pareto_front.png").is_file()
    assert (tmp_path / "samapy_outputs" / "data" / "Outputforplotting.csv").is_file()
