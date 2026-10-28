"""Reproducible convergence tests for the four optimizers (issue #10).

Each optimizer runs on benchmark functions with a known global minimum of 0,
using the same code path that optimizes the SAMA design problem; only the
objective and the bounds are replaced.

Problems
    Shifted sphere, 5-D, bounds [-5, 5]^5, minimum 0 at
        x* = (1.0, -2.0, 0.5, 3.0, -1.5). The shift keeps the optimum away
        from the centre of the box, so centre-biased initialization cannot
        find it by construction.
    Rosenbrock, 2-D, bounds [-2, 2]^2, minimum 0 at x* = (1, 1). A curved,
        non-separable valley.
    Rosenbrock, 5-D, bounds [-2, 2]^5, minimum 0 at x* = (1, ..., 1).

Budget
    Population 30, 300 iterations, 1 run: about 30 x 301 = 9,030 objective
    evaluations (ADE and ABC use a few more for local search and scouts;
    the exact count is checked below).

Seeds and pass criteria
    Seeds 0-9 (NumPy and Numba are both seeded, so every run is exactly
    reproducible). A run succeeds when its best objective value is below the
    tolerance. A test passes when at least 9 of the 10 runs succeed, so a
    single unlucky seed does not fail the suite while systematic failure does.

Tolerances
    1e-2 for the shifted sphere and 2-D Rosenbrock. For the sphere this means
    the best point is within a Euclidean distance of 0.1 of x*, i.e. 1% of
    the width of the search box. The worst run observed during development
    was 1.9e-3 (ABC on 2-D Rosenbrock), so the tolerance has a margin of
    more than 5x.
    5-D Rosenbrock is checked only for ADE, with tolerance 1e-6 (worst run
    observed 2.1e-8). With this budget PSO, ABC and GWO reached best values
    between 0.1 and 3 on this problem, so a meaningful common tolerance cannot
    be set for them at this budget; this is documented rather than hidden.

Every run must also return a finite result inside the bounds.
"""
import numpy as np
import pytest

from samapy.optimizers.swarm import Swarm
from samapy.optimizers.AdvancedDifferentialEvolution import AdvancedDifferentialEvolution
from samapy.optimizers.ArtificialBeeColony import ImprovedArtificialBeeColony
from samapy.optimizers.GreyWolfOptimizer import GreyWolfOptimizer

OPTIMIZERS = {
    "pso": Swarm,
    "ade": AdvancedDifferentialEvolution,
    "abc": ImprovedArtificialBeeColony,
    "gwo": GreyWolfOptimizer,
}

SHIFT = np.array([1.0, -2.0, 0.5, 3.0, -1.5])


def shifted_sphere(x):
    x = np.asarray(x, dtype=float)
    return float(np.sum((x - SHIFT) ** 2))


def rosenbrock(x):
    x = np.asarray(x, dtype=float)
    return float(np.sum(100.0 * (x[1:] - x[:-1] ** 2) ** 2 + (1.0 - x[:-1]) ** 2))


PROBLEMS = {
    "sphere5": dict(f=shifted_sphere, lo=[-5.0] * 5, hi=[5.0] * 5, x_star=SHIFT, tol=1e-2),
    "rosenbrock2": dict(f=rosenbrock, lo=[-2.0] * 2, hi=[2.0] * 2, x_star=np.ones(2), tol=1e-2),
    "rosenbrock5": dict(f=rosenbrock, lo=[-2.0] * 5, hi=[2.0] * 5, x_star=np.ones(5), tol=1e-6),
}

N_POP, MAX_IT = 30, 300
SEEDS = range(10)
MIN_SUCCESSES = 9
# Nominal budget plus a 5% allowance for ADE local search and ABC scout bees.
MAX_EVALUATIONS = int(1.05 * N_POP * (MAX_IT + 1))

CASES = [(a, p) for p in ("sphere5", "rosenbrock2") for a in OPTIMIZERS] + [("ade", "rosenbrock5")]


def run(algo, problem, seed, n_pop=N_POP, max_it=MAX_IT):
    spec = PROBLEMS[problem]
    calls = {"n": 0}

    def objective(x):
        calls["n"] += 1
        return spec["f"](x)

    opt = OPTIMIZERS[algo](cost_function=objective, var_min=spec["lo"], var_max=spec["hi"],
                           n_pop=n_pop, max_it=max_it, run_time=1, seed=seed,
                           generate_results=False, verbose=False)
    X, best = opt.optimize()
    return np.asarray(X, dtype=float), float(best), calls["n"]


@pytest.mark.parametrize("algo,problem", CASES)
def test_converges_to_known_optimum(algo, problem):
    spec = PROBLEMS[problem]
    lo, hi = np.asarray(spec["lo"]), np.asarray(spec["hi"])
    successes, results = 0, []
    for seed in SEEDS:
        X, best, n_evals = run(algo, problem, seed)

        assert np.all(np.isfinite(X)) and np.isfinite(best), f"seed {seed}: non-finite result"
        assert np.all(X >= lo) and np.all(X <= hi), f"seed {seed}: result outside bounds {X}"
        assert n_evals <= MAX_EVALUATIONS, f"seed {seed}: {n_evals} evaluations > {MAX_EVALUATIONS}"
        # the reported best value belongs to the reported point
        assert best == pytest.approx(spec["f"](X), rel=1e-12, abs=1e-15)

        results.append(best)
        successes += best < spec["tol"]

    assert successes >= MIN_SUCCESSES, (
        f"{algo} on {problem}: {successes}/{len(SEEDS)} runs below {spec['tol']}; "
        f"best values {['%.2e' % r for r in results]}")


@pytest.mark.parametrize("algo", sorted(OPTIMIZERS))
def test_same_seed_gives_identical_result(algo):
    a = run(algo, "rosenbrock2", seed=3, max_it=40)
    b = run(algo, "rosenbrock2", seed=3, max_it=40)
    np.testing.assert_array_equal(a[0], b[0])
    assert a[1] == b[1]


@pytest.mark.parametrize("algo", sorted(OPTIMIZERS))
def test_more_iterations_do_not_give_a_worse_result(algo):
    # With the same seed, the best value found can only improve or stay equal
    # as the run continues (all four optimizers keep the best-so-far solution).
    spec = PROBLEMS["sphere5"]
    opt = OPTIMIZERS[algo](cost_function=spec["f"], var_min=spec["lo"], var_max=spec["hi"],
                           n_pop=N_POP, max_it=100, run_time=1, seed=0,
                           generate_results=False, verbose=False)
    opt.optimize()
    curve = np.asarray(opt.solution_cost_curve[0])
    assert len(curve) == 100
    assert np.all(np.diff(curve) <= 0), "best-so-far curve increased"
    assert curve[-1] < curve[0]
