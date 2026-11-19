"""Compare SAMA's Pareto front with weighted-sum (scalarized) optimization.

For one SAMA configuration, this script

1. runs the multi-objective mode (NSGA-II) with the objectives, population,
   generations and seed of the configuration, exactly as
   ``samapy-run -c <config> --mode multi`` does, and measures how far each
   design on the front lies above the convex hull of the front;
2. runs SAMA's single-objective ADE optimizer once for each of a set of
   evenly spaced weight vectors, each time minimizing the weighted sum of the
   same objectives (normalized over the front) under the same constraints,
   as a weighted-sum sweep would;
3. reports which designs the weighted-sum runs return, and writes a figure
   and two CSV files.

A design on the front is supported when some weighted sum of the objectives
is minimized by it among the designs found, and unsupported when, for every
choice of weights, another design on the front has a lower weighted sum
(its hull gap is that margin, as a fraction of the objective ranges). An
unsupported design cannot be the result of any weighted-sum optimization,
however the weights are chosen; it is found only by searching for the Pareto
front itself. As in samapy-run, designs less than 1% of the ranges above the
convex hull count as supported, because an approximate front is not exactly
convex even where the true front is.

Each weight vector gives every objective a weight of at least 1%, so that
each run returns a Pareto-optimal design: with a weight of exactly zero an
objective would be left unconstrained and the run could return any design
that is best in the others.

Usage, from the repository root:

    python examples/compare_weighted_sum.py -c examples/moo_grid_no_export.yaml --output moo_grid_no_export_comparison
    python examples/compare_weighted_sum.py -c examples/moo_offgrid.yaml --output moo_offgrid_comparison

Options: --seed (default: seed in the configuration, else 1) and
--divisions (default 10: 11 weight vectors for two objectives, 66 for
three). Each weighted-sum run uses the configuration's nPop and MaxIt. With
the example configurations the script takes about 15 minutes for the
grid-connected house and 5 minutes for the off-grid system.
"""
import argparse
import contextlib
import io
import itertools
import sys
import time
import warnings
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Use the source tree when the script is run from a repository checkout.
_SRC = Path(__file__).resolve().parent.parent / "src"
if (_SRC / "samapy").is_dir():
    sys.path.insert(0, str(_SRC))

from samapy.cli.config_loader import apply_config, load_config  # noqa: E402
from samapy.cli.runner import _patch_fitness, _patch_results  # noqa: E402

MIN_WEIGHT = 0.01   # smallest weight of any objective


def weight_vectors(n_obj, divisions):
    """Evenly spaced weight vectors on the simplex, each objective weighted at least 1%."""
    lattice = [np.array(c, dtype=float) / divisions
               for c in itertools.product(range(divisions + 1), repeat=n_obj) if sum(c) == divisions]
    lattice.sort(key=lambda w: tuple(-w))
    return [MIN_WEIGHT + (1 - n_obj * MIN_WEIGHT) * w for w in lattice]


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("-c", "--config", required=True, help="SAMA YAML configuration")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--divisions", type=int, default=10)
    parser.add_argument("--output", default="weighted_sum_comparison")
    args = parser.parse_args()
    # A design with no supply at all (every size zero) has undefined LEM and RE;
    # the optimizers discard it, so the division warnings are not shown.
    warnings.filterwarnings("ignore", message="invalid value encountered", category=RuntimeWarning)

    config = load_config(args.config)
    with contextlib.redirect_stdout(io.StringIO()):
        indata = apply_config(config)
    _patch_fitness(indata)
    _patch_results(indata)
    seed = args.seed if args.seed is not None else int(config.get("seed", 1))

    from samapy.core.Fitness import evaluate
    from samapy.optimizers.AdvancedDifferentialEvolution import AdvancedDifferentialEvolution
    from samapy.optimizers.multiobjective import OBJECTIVES, VARIABLES as DESIGN, MultiObjectiveOptimizer

    objectives = list(indata.MOO_objectives)
    signs = np.array([OBJECTIVES[o][1] for o in objectives])
    labels = [OBJECTIVES[o][2] for o in objectives]
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)

    # 1. Pareto front, as samapy-run --mode multi computes it
    t0 = time.time()
    moo = MultiObjectiveOptimizer(seed=seed, generate_results=False, verbose=False)
    front = moo.optimize()
    F_front = front.front_F * signs                 # as minimized
    ideal, nadir = F_front.min(axis=0), F_front.max(axis=0)
    span = np.where(nadir - ideal > 0, nadir - ideal, 1.0)
    G_front = (F_front - ideal) / span
    unsupported = ~front.supported
    print(f"Configuration : {args.config}")
    print(f"Objectives    : {', '.join(objectives)}  (seed {seed})")
    print(f"Pareto front  : {len(F_front)} designs from one NSGA-II run of {moo.n_pop} x {moo.n_gen} "
          f"evaluations ({time.time() - t0:.0f} s)")
    print(f"Hull gap      : largest {front.hull_gap.max():.1%} of the objective ranges; "
          f"{int(unsupported.sum())} designs more than 1% above the convex hull (unsupported)")
    if unsupported.any():
        for k, o in enumerate(objectives):
            values = front.front_F[unsupported, k]
            print(f"                unsupported designs span {o} = {values.min():.6g} to {values.max():.6g}")

    # 2. One weighted-sum optimization per weight vector
    weights = weight_vectors(len(objectives), args.divisions)

    def weighted_sum(w):
        def cost(x):
            m = evaluate(x)
            f = signs * np.array([m[OBJECTIVES[o][0]] for o in objectives])
            violation = sum(max(g, 0.0) for g in m["constraints"].values())
            value = float(np.dot(w, (f - ideal) / span) + 1e3 * violation)
            return value if np.isfinite(value) else 1e9
        return cost

    rows = []
    for k, w in enumerate(weights):
        t1 = time.time()
        x, _ = AdvancedDifferentialEvolution(cost_function=weighted_sum(w), seed=seed + k,
                                             generate_results=False, verbose=False).optimize()
        m = evaluate(x)
        f = signs * np.array([m[OBJECTIVES[o][0]] for o in objectives])
        g = (f - ideal) / span
        dist = np.linalg.norm(G_front - g, axis=1)
        nearest = int(np.argmin(dist))
        dominated = bool(np.any(np.all(G_front <= g + 1e-3, axis=1) & np.any(G_front < g - 1e-3, axis=1)))
        rows.append(dict(weights=" ".join(f"{v:.3f}" for v in w),
                         **{o: m[OBJECTIVES[o][0]] for o in objectives},
                         **dict(zip(DESIGN, x)),
                         feasible=all(v <= 1e-9 for v in m["constraints"].values()),
                         dominated_by_front=dominated, nearest_front_design=nearest,
                         nearest_is_supported=bool(front.supported[nearest]),
                         distance_to_nearest=float(dist[nearest])))
        print(f"  weights ({rows[-1]['weights']}): "
              + ", ".join(f"{o} = {m[OBJECTIVES[o][0]]:.6g}" for o in objectives)
              + f"  [nearest front design {nearest}, "
                f"{'supported' if front.supported[nearest] else 'UNSUPPORTED'}, "
                f"distance {dist[nearest]:.3f}]  ({time.time() - t1:.0f} s)")
    ws = pd.DataFrame(rows)

    # 3. Summary
    close = ws.distance_to_nearest <= 0.02
    reached = set(ws.loc[close, "nearest_front_design"])
    reached_unsupported = sorted(i for i in reached if not front.supported[i])
    distinct = len(ws.round({o: 6 for o in objectives}).drop_duplicates(subset=objectives))
    print(f"\nWeighted-sum sweep : {len(ws)} runs of {indata.nPop} x {indata.MaxIt} evaluations, "
          f"{distinct} distinct designs, {int(ws.feasible.sum())} feasible")
    print(f"Front designs that a weighted-sum run reached (within 2% of the ranges): {len(reached)} of "
          f"{len(F_front)}; unsupported designs among them: {len(reached_unsupported)} of {int(unsupported.sum())}")
    print(f"Weighted-sum designs dominated by a design on the front: {int(ws.dominated_by_front.sum())}")

    front_df = pd.DataFrame(front.front_F, columns=objectives)
    front_df["supported"] = front.supported
    front_df["hull_gap"] = front.hull_gap
    for j, name in enumerate(DESIGN):
        front_df[name] = front.front_X[:, j]
    front_df.to_csv(out / "pareto_front_classified.csv", index_label="front_design")
    ws.to_csv(out / "weighted_sum_runs.csv", index=False)

    # Figure: each pair of objectives
    pairs = list(itertools.combinations(range(len(objectives)), 2))
    fig, axes = plt.subplots(1, len(pairs), figsize=(6.0 * len(pairs), 4.6), squeeze=False)
    sup = front.supported
    for ax, (i, j) in zip(axes[0], pairs):
        if len(objectives) == 2:
            hull = np.flatnonzero(front.hull_gap <= 1e-9)
            hull = hull[np.argsort(front.front_F[hull, i])]
            ax.plot(front.front_F[hull, i], front.front_F[hull, j], color="0.5", lw=0.8,
                    label="convex hull of the front")
        ax.scatter(front.front_F[sup, i], front.front_F[sup, j], s=22, color="C0",
                   label="Pareto front (NSGA-II): supported")
        if (~sup).any():
            ax.scatter(front.front_F[~sup, i], front.front_F[~sup, j], s=26, facecolors="none",
                       edgecolors="C3", linewidths=1.2,
                       label="Pareto front (NSGA-II): unsupported,\nno weighted sum selects it")
        ax.scatter(ws[objectives[i]], ws[objectives[j]], s=60, marker="x", color="k",
                   label=f"weighted-sum sweep (ADE), {len(ws)} runs")
        ax.set_xlabel(labels[i])
        ax.set_ylabel(labels[j])
        ax.grid(True, alpha=0.3)
    axes[0][0].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out / "pareto_vs_weighted_sum.png", dpi=300)
    plt.close(fig)
    print(f"\nSaved: {out / 'pareto_front_classified.csv'}, {out / 'weighted_sum_runs.csv'}, "
          f"{out / 'pareto_vs_weighted_sum.png'}")
    print(f"Total time: {time.time() - t0:.0f} s")


if __name__ == "__main__":
    main()
