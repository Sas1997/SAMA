"""Multi-objective design optimization for SAMA.

The single-objective optimizers (PSO, ADE, ABC, GWO) minimize one scalar
fitness value. This module instead searches for the set of Pareto-optimal
designs, i.e. the designs for which no objective can be improved without
worsening another, using the NSGA-II algorithm from the pymoo library
(Blank and Deb, 2020).

Objectives (any two or more):
    npc    net present cost                      (minimized)
    lcoe   levelized cost of energy              (minimized)
    lem    levelized emissions, kg/kWh           (minimized)
    lpsp   loss of power supply probability      (minimized)
    re     renewable energy fraction             (maximized)

Constraints are handled as constraints, not as penalties: a design is
feasible when it meets the reliability limit (LPSP_max), the minimum
renewable fraction (RE_min), the budget and, where configured, the roof
area, PV capacity, generation cap and net-metering conditions. Objectives
are traded off within those limits, so to explore, for example, cost
against reliability, relax LPSP_max and choose 'lpsp' as an objective.

One design from the front is then selected for the detailed SAMA results
(figures, cash flow and hourly data). The selection rule is MOO_selection:
    'compromise'   the design closest to the ideal point with equal weight on
                   every objective (achievement scalarizing function on the
                   objectives normalized over the front)
    'min_<name>'   the design that is best in one objective, e.g. 'min_npc'
                   (for 're' this is the design with the highest fraction)
    [w1, w2, ...]  the same achievement function with one weight per
                   objective; a larger weight favours that objective

Example:
    from samapy.optimizers.multiobjective import MultiObjectiveOptimizer
    result = MultiObjectiveOptimizer(objectives=['npc', 're'], seed=1).optimize()
    result.front_F      # objective values of the Pareto-optimal designs
    result.selected_X   # design used for the detailed results
"""
import os
import sys
from dataclasses import dataclass, field

import matplotlib
import numpy as np

_is_headless = (sys.platform == 'linux' and not os.environ.get('DISPLAY', ''))
if _is_headless:
    matplotlib.use('Agg')

import matplotlib.pyplot as plt
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.core.problem import ElementwiseProblem
from pymoo.optimize import minimize

from samapy.core.Input_Data import InData
from samapy.optimizers._common import resolve_settings

# name -> (key in the evaluate() result, +1 minimize / -1 maximize, label,
#          smallest difference reported as a trade-off: (absolute, relative))
OBJECTIVES = {
    'npc': ('NPC', 1.0, 'Net present cost', (0.0, 1e-4)),
    'lcoe': ('LCOE', 1.0, 'Levelized cost of energy', (0.0, 1e-4)),
    'lem': ('LEM', 1.0, 'Levelized emissions [kg/kWh]', (1e-5, 0.0)),
    'lpsp': ('LPSP', 1.0, 'Loss of power supply probability', (1e-7, 0.0)),
    're': ('RE', -1.0, 'Renewable energy fraction', (1e-3, 0.0)),
}
VARIABLES = ['PV_modules', 'wind_turbines', 'batteries', 'generator_kW', 'inverter_kW']


def _default_evaluate(x):
    from samapy.core.Fitness import evaluate
    return evaluate(x)


@dataclass
class ParetoResult:
    """Pareto-optimal designs found by the multi-objective optimizer."""
    objectives: list
    front_X: np.ndarray            # one row per Pareto-optimal design
    front_F: np.ndarray            # objective values in natural units (RE as a fraction, not negated)
    front_metrics: list            # full evaluate() result for each design
    selected_index: int
    feasible: bool                 # False if no design met all constraints
    constraint_names: list = field(default_factory=list)

    @property
    def selected_X(self):
        return self.front_X[self.selected_index]

    @property
    def selected_F(self):
        return self.front_F[self.selected_index]


class _SAMAProblem(ElementwiseProblem):
    def __init__(self, objectives, evaluate, xl, xu, constraint_names):
        self._objectives = objectives
        self._evaluate_design = evaluate
        self._constraint_names = constraint_names
        super().__init__(n_var=len(xl), n_obj=len(objectives),
                         n_ieq_constr=len(constraint_names), xl=xl, xu=xu)

    def _evaluate(self, x, out, *args, **kwargs):
        m = self._evaluate_design(x)
        out["F"] = [OBJECTIVES[o][1] * m[OBJECTIVES[o][0]] for o in self._objectives]
        if self._constraint_names:
            out["G"] = [m["constraints"][c] for c in self._constraint_names]


def _epsilon_nondominated(F_min, epsilon, resolution=None):
    """Indices of the designs kept when reporting the front.

    A design is dropped when another design is no worse than it by more than
    a tolerance in every objective and better by more than the tolerance in
    at least one (additive epsilon-dominance). The tolerance is epsilon times
    the range of each objective over the front, so differences far below any
    practical significance (for example a renewable fraction that improves in
    the sixth decimal at a large extra cost) are not reported as trade-offs.
    Designs within the tolerance of an already kept design in every objective
    are merged. The tolerance is never smaller than the objective's
    resolution (OBJECTIVES, e.g. 0.1 percentage point of renewable fraction,
    0.01% of net present cost). With epsilon = 0 and no resolution this keeps
    the exact non-dominated set without duplicates.
    """
    F_min = np.asarray(F_min, dtype=float)
    tol = epsilon * (F_min.max(axis=0) - F_min.min(axis=0))
    if resolution is not None:
        tol = np.maximum(tol, np.asarray(resolution, dtype=float))
    keep = []
    for i in range(len(F_min)):
        better_or_close = np.all(F_min <= F_min[i] + tol, axis=1)
        clearly_better = np.any(F_min < F_min[i] - tol, axis=1)
        if np.any(better_or_close & clearly_better):
            continue
        if any(np.all(np.abs(F_min[j] - F_min[i]) <= tol) for j in keep):
            continue
        keep.append(i)
    return keep or [int(np.argmin(F_min[:, 0]))]


def select_from_front(F_min, objectives, selection):
    """Index of the design chosen from a front given as minimization values.

    F_min holds the objective values as minimized (RE negated).
    """
    F_min = np.asarray(F_min, dtype=float)
    if len(F_min) == 1:
        return 0
    if isinstance(selection, str) and selection.startswith('min_'):
        name = selection[4:]
        if name not in objectives:
            raise ValueError(f"MOO_selection '{selection}': '{name}' is not one of the objectives {objectives}")
        return int(np.argmin(F_min[:, objectives.index(name)]))
    if isinstance(selection, str):
        if selection != 'compromise':
            raise ValueError(f"unknown MOO_selection '{selection}'")
        weights = np.ones(len(objectives))
    else:
        weights = np.asarray(selection, dtype=float)
        if weights.shape != (len(objectives),) or np.any(weights <= 0):
            raise ValueError("MOO_selection weights need one positive value per objective")
    ideal, nadir = F_min.min(axis=0), F_min.max(axis=0)
    span = np.where(nadir - ideal > 0, nadir - ideal, 1.0)
    normalized = (F_min - ideal) / span
    # Achievement scalarizing function: a larger weight pulls the choice
    # towards the ideal value of that objective.
    asf = np.max(normalized * weights, axis=1)
    return int(np.argmin(asf))


class MultiObjectiveOptimizer:
    """NSGA-II search for Pareto-optimal SAMA designs.

    Arguments (all optional; defaults come from InData when constructed):
        objectives        list of names from OBJECTIVES (default InData.MOO_objectives)
        n_pop, n_gen      population size and generations (default MOO_n_pop/MOO_n_gen,
                          falling back to nPop and MaxIt)
        selection         rule for the design used for the detailed results
                          (default InData.MOO_selection)
        seed              integer seed for reproducible runs
        var_min, var_max  bounds of the design variables (default: SAMA bounds)
        evaluate          function returning objectives and constraints for a design
                          (default: samapy.core.Fitness.evaluate)
        epsilon           resolution for reporting the front, as a fraction of each
                          objective's range (default 1e-3), and never finer than the
                          objective's own resolution (see OBJECTIVES); 0 reports the
                          exact non-dominated set
        generate_results  False skips the CSV, the plot and Gen_Results
        output_dir        folder for pareto_front.csv/.png (default: SAMA output folder)
        verbose           print progress
    """

    def __init__(self, objectives=None, n_pop=None, n_gen=None, selection=None, seed=None,
                 var_min=None, var_max=None, evaluate=None, epsilon=1e-3,
                 generate_results=True, output_dir=None, verbose=True):
        objectives = list(InData.MOO_objectives if objectives is None else objectives)
        unknown = [o for o in objectives if o not in OBJECTIVES]
        if unknown:
            raise ValueError(f"unknown objective(s) {unknown}; choose from {sorted(OBJECTIVES)}")
        if len(objectives) < 2 or len(set(objectives)) != len(objectives):
            raise ValueError("choose at least two different objectives")
        self.objectives = objectives

        n_pop = n_pop if n_pop is not None else getattr(InData, 'MOO_n_pop', None)
        n_gen = n_gen if n_gen is not None else getattr(InData, 'MOO_n_gen', None)
        (self.VarMin, self.VarMax, self.n_pop, self.n_gen,
         _) = resolve_settings(InData, var_min, var_max, n_pop, n_gen, 1)
        self.selection = InData.MOO_selection if selection is None else selection
        self.seed = seed
        if not 0 <= epsilon < 1:
            raise ValueError("epsilon must be in [0, 1)")
        self.epsilon = epsilon
        self.evaluate = evaluate or _default_evaluate
        self.generate_results = generate_results
        self.output_dir = output_dir
        self.verbose = verbose

    def optimize(self):
        """Run NSGA-II and return a ParetoResult."""
        xl = np.asarray(self.VarMin, dtype=float)
        xu = np.asarray(self.VarMax, dtype=float)
        # The set of constraints depends on the configuration (e.g. net metering).
        constraint_names = sorted(self.evaluate((xl + xu) / 2)["constraints"])
        problem = _SAMAProblem(self.objectives, self.evaluate, xl, xu, constraint_names)
        algorithm = NSGA2(pop_size=self.n_pop)
        res = minimize(problem, algorithm, ('n_gen', self.n_gen), seed=self.seed,
                       verbose=self.verbose, save_history=False)

        if res.X is not None:
            X = np.atleast_2d(res.X)
            feasible = True
        else:
            # No feasible design found: report the least-violating designs.
            cv = res.pop.get("CV").ravel()
            X = np.atleast_2d(res.pop.get("X")[cv == cv.min()])
            feasible = False
            print("⚠️  No design met all constraints; showing the least-violating designs. "
                  "Consider relaxing the constraints or increasing n_gen.")

        metrics = [self.evaluate(x) for x in X]
        F_min = np.array([[OBJECTIVES[o][1] * m[OBJECTIVES[o][0]] for o in self.objectives] for m in metrics])
        resolution = None
        if self.epsilon > 0:
            resolution = [max(OBJECTIVES[o][3][0], OBJECTIVES[o][3][1] * np.median(np.abs(F_min[:, k])))
                          for k, o in enumerate(self.objectives)]
        keep = _epsilon_nondominated(F_min, self.epsilon, resolution)
        X, F_min, metrics = X[keep], F_min[keep], [metrics[i] for i in keep]
        order = np.argsort(F_min[:, 0], kind='stable')
        X, F_min, metrics = X[order], F_min[order], [metrics[i] for i in order]
        signs = np.array([OBJECTIVES[o][1] for o in self.objectives])
        result = ParetoResult(objectives=self.objectives, front_X=X, front_F=F_min * signs,
                              front_metrics=metrics,
                              selected_index=select_from_front(F_min, self.objectives, self.selection),
                              feasible=feasible, constraint_names=constraint_names)
        if self.verbose:
            print(f"Pareto front: {len(X)} designs; selected ({self.selection}): "
                  + ", ".join(f"{o} = {v:.6g}" for o, v in zip(self.objectives, result.selected_F)))
        if self.generate_results:
            self._write_outputs(result)
        return result

    def _write_outputs(self, result):
        import pandas as pd
        import samapy.results.Results as Results

        out = self.output_dir or Results.OUTPUT_DIR
        os.makedirs(out, exist_ok=True)
        rows = []
        for i, (x, m) in enumerate(zip(result.front_X, result.front_metrics)):
            row = dict(zip(VARIABLES, x))
            row.update({k: m[k] for k in ('NPC', 'LCOE', 'LEM', 'LPSP', 'RE', 'I_Cost')})
            row['selected'] = i == result.selected_index
            rows.append(row)
        csv_path = os.path.join(out, 'pareto_front.csv')
        pd.DataFrame(rows).to_csv(csv_path, index=False)
        print(f"✓ Saved: {csv_path}")

        n = len(self.objectives)
        pairs = [(i, j) for i in range(n) for j in range(i + 1, n)]
        fig, axes = plt.subplots(1, len(pairs), figsize=(5 * len(pairs), 4), squeeze=False)
        for ax, (i, j) in zip(axes[0], pairs):
            ax.scatter(result.front_F[:, i], result.front_F[:, j], s=18, label='Pareto-optimal designs')
            ax.scatter(*result.selected_F[[i, j]], s=80, marker='*', color='red',
                       label=f'selected ({self.selection})')
            ax.set_xlabel(OBJECTIVES[self.objectives[i]][2])
            ax.set_ylabel(OBJECTIVES[self.objectives[j]][2])
            ax.grid(True, alpha=0.3)
        axes[0][0].legend()
        fig.tight_layout()
        png_path = os.path.join(out, 'pareto_front.png')
        fig.savefig(png_path, dpi=300)
        plt.close(fig)
        print(f"✓ Saved: {png_path}")

        if self.output_dir:
            Results.Gen_Results(result.selected_X, output_dir=self.output_dir)
        else:
            Results.Gen_Results(result.selected_X)
