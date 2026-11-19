"""The multi-objective examples in examples/ and the weighted-sum comparison script.

The fast tests check that the example configurations are valid and set up the
documented cases; the slow tests run them with a small population.
"""
import re
import subprocess
import sys
from pathlib import Path

import pandas as pd
import pytest
import yaml

ROOT = Path(__file__).resolve().parent.parent
EXAMPLES = ROOT / "examples"
GRID = EXAMPLES / "moo_grid_no_export.yaml"
OFFGRID = EXAMPLES / "moo_offgrid.yaml"
REFERENCE = EXAMPLES / "reference_config.yaml"
COMPARE = EXAMPLES / "compare_weighted_sum.py"

pytestmark = pytest.mark.skipif(not GRID.is_file(), reason="examples folder not found")

RUN = [sys.executable, "-c", "from samapy.cli.runner import main; main()"]


def load(path):
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def changed_keys(path):
    ref, cfg = load(REFERENCE), load(path)
    return {k: cfg[k] for k in cfg if cfg[k] != ref.get(k)}


def test_grid_example_differs_from_the_reference_case_only_as_documented():
    assert changed_keys(GRID) == {
        "optimization_mode": "multi", "MOO_n_pop": 100, "MOO_n_gen": 200, "seed": 1,
        "NEM": 0, "cap_option": 4, "Csell_flat": 0.0, "RE_min_rate": 0.0, "RE_min": 0.0}
    assert load(GRID)["MOO_objectives"] == ["npc", "re"]


def test_offgrid_example_differs_from_the_reference_case_only_as_documented():
    assert changed_keys(OFFGRID) == {
        "optimization_mode": "multi", "MOO_objectives": ["npc", "lpsp"], "MOO_n_pop": 100,
        "MOO_n_gen": 200, "seed": 1, "Grid": 0, "NEM": 0, "EV": 0, "HP": 0, "WT": 1, "cap_option": 4,
        "RE_min_rate": 0.0, "RE_min": 0.0, "LPSP_max_rate": 5.0, "LPSP_max": 0.05}


@pytest.mark.parametrize("config", [GRID, OFFGRID], ids=["grid", "offgrid"])
def test_example_configuration_is_valid(config, tmp_path):
    r = subprocess.run(RUN + ["-c", str(config), "--dry-run", "--verbose"], cwd=tmp_path,
                       capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=900)
    assert r.returncode == 0, r.stdout[-3000:] + r.stderr[-3000:]
    assert "multi-objective (NSGA-II)" in r.stdout and "Seed        : 1" in r.stdout


def small(config, tmp_path, n_pop=12, n_gen=6, n_pop_single=8, max_it=4):
    text = config.read_text(encoding="utf-8")
    for key, value in (("MOO_n_pop", n_pop), ("MOO_n_gen", n_gen), ("nPop", n_pop_single), ("MaxIt", max_it)):
        text = re.sub(rf"(?m)^{key}: \d+", f"{key}: {value}", text)
    path = tmp_path / config.name
    path.write_text(text, encoding="utf-8")
    return path


@pytest.mark.slow
@pytest.mark.parametrize("config", [GRID, OFFGRID], ids=["grid", "offgrid"])
def test_example_runs_in_multi_objective_mode(config, tmp_path):
    path = small(config, tmp_path)
    r = subprocess.run(RUN + ["-c", str(path), "--no-gui", "--output", "out"], cwd=tmp_path,
                       capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=1800)
    assert r.returncode == 0, r.stdout[-3000:] + r.stderr[-3000:]
    front = pd.read_csv(tmp_path / "out" / "pareto_front.csv")
    assert len(front) >= 1 and front["selected"].sum() == 1 and "supported" in front


@pytest.mark.slow
def test_weighted_sum_comparison_script(tmp_path):
    path = small(OFFGRID, tmp_path)
    r = subprocess.run([sys.executable, str(COMPARE), "-c", str(path), "--divisions", "1",
                        "--output", "cmp"], cwd=tmp_path, capture_output=True, text=True,
                       encoding="utf-8", errors="replace", timeout=1800)
    assert r.returncode == 0, r.stdout[-3000:] + r.stderr[-3000:]
    runs = pd.read_csv(tmp_path / "cmp" / "weighted_sum_runs.csv")
    assert len(runs) == 2                      # the two ends of the weight range
    front = pd.read_csv(tmp_path / "cmp" / "pareto_front_classified.csv")
    assert {"npc", "lpsp", "supported", "hull_gap"} <= set(front)
    assert (tmp_path / "cmp" / "pareto_vs_weighted_sum.png").is_file()


FRONT = r"""
import contextlib, io, json, sys, warnings
warnings.filterwarnings("ignore")
from samapy.cli.config_loader import load_config, apply_config
from samapy.cli import runner
with contextlib.redirect_stdout(io.StringIO()):
    d = apply_config(load_config(sys.argv[1]))
runner._patch_fitness(d)
from samapy.optimizers.multiobjective import MultiObjectiveOptimizer
r = MultiObjectiveOptimizer(seed=1, generate_results=False, verbose=False).optimize()
u = ~r.supported
print(json.dumps({"n": len(r.front_F), "max_gap": float(r.hull_gap.max()), "unsupported": int(u.sum()),
                  "F": r.front_F.tolist(), "F_unsupported": r.front_F[u].tolist()}))
"""


def front_of(config, tmp_path):
    r = subprocess.run([sys.executable, "-c", FRONT, str(config)], cwd=tmp_path, capture_output=True,
                       text=True, encoding="utf-8", errors="replace", timeout=1800)
    assert r.returncode == 0, r.stderr[-3000:]
    import json
    return json.loads(r.stdout.strip().splitlines()[-1])


@pytest.mark.slow
def test_offgrid_example_front_has_a_non_convex_stretch(tmp_path):
    """Documented in Section 7.6.3: designs no weighted sum can return."""
    f = front_of(OFFGRID, tmp_path)
    assert f["n"] >= 50
    assert f["max_gap"] > 0.03 and f["unsupported"] >= 8
    for npc, lpsp in f["F_unsupported"]:
        assert 0.002 < lpsp < 0.025
    lpsp = [p for _, p in f["F"]]
    assert max(lpsp) > 0.045 and min(lpsp) < 1e-4


@pytest.mark.slow
def test_grid_example_front_is_convex(tmp_path):
    f = front_of(GRID, tmp_path)
    assert f["n"] >= 30 and f["unsupported"] == 0 and f["max_gap"] < 0.01
    npc = [c for c, _ in f["F"]]
    re_fraction = [r for _, r in f["F"]]
    assert min(npc) < 87200 and max(re_fraction) > 0.89 and min(re_fraction) < 0.55
