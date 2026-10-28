import sys
import matplotlib
import os
# Agg only on headless Linux (servers, CI, Docker).
# Never force Agg on Windows or macOS — both have native GUI displays.
_is_headless = (sys.platform == 'linux' and not os.environ.get('DISPLAY', ''))
if _is_headless:
    matplotlib.use('Agg')

import matplotlib.pyplot as plt
from samapy.results.Results import Gen_Results
# Loading Inputs
from samapy.core.Input_Data import InData
import numpy as np
from time import process_time
from numba import njit
import multiprocessing as mp

start = process_time()

PV = InData.PV
WT = InData.WT
Bat = InData.Bat
DG = InData.DG
Run_Time = InData.Run_Time
nPop = InData.nPop
MaxIt = InData.MaxIt
VarMin = InData.VarMin
VarMax = InData.VarMax

# Problem Definition
from samapy.core.Fitness import fitness as cost_function
from samapy.optimizers._common import seed_all, resolve_settings


# Vectorized fitness evaluation
def evaluate_population(positions):
    """Vectorized fitness evaluation for entire population"""
    return np.array([cost_function(pos) for pos in positions])


# JIT-compiled position update function
@njit
def update_positions_vectorized(positions, alpha_pos, beta_pos, delta_pos, a, VarMin, VarMax):
    """Vectorized position update with JIT compilation"""
    nPop, nVar = positions.shape
    new_positions = np.zeros_like(positions)

    for i in range(nPop):
        for j in range(nVar):
            # Generate random numbers
            r1_1, r2_1 = np.random.random(), np.random.random()
            r1_2, r2_2 = np.random.random(), np.random.random()
            r1_3, r2_3 = np.random.random(), np.random.random()

            # Calculate A and C parameters
            A1 = 2 * a * r1_1 - a
            C1 = 2 * r2_1
            A2 = 2 * a * r1_2 - a
            C2 = 2 * r2_2
            A3 = 2 * a * r1_3 - a
            C3 = 2 * r2_3

            # Calculate distances and new positions
            D_alpha = abs(C1 * alpha_pos[j] - positions[i][j])
            X1 = alpha_pos[j] - A1 * D_alpha

            D_beta = abs(C2 * beta_pos[j] - positions[i][j])
            X2 = beta_pos[j] - A2 * D_beta

            D_delta = abs(C3 * delta_pos[j] - positions[i][j])
            X3 = delta_pos[j] - A3 * D_delta

            # Update position
            new_positions[i][j] = (X1 + X2 + X3) / 3

            # Apply bounds
            if new_positions[i][j] < VarMin[j]:
                new_positions[i][j] = VarMin[j]
            elif new_positions[i][j] > VarMax[j]:
                new_positions[i][j] = VarMax[j]

    return new_positions


@njit
def update_leaders_vectorized(fitness_values, positions, alpha_pos, beta_pos, delta_pos,
                              alpha_score, beta_score, delta_score):
    """Vectorized leader update with JIT compilation"""
    nPop = len(fitness_values)
    new_alpha_pos = alpha_pos.copy()
    new_beta_pos = beta_pos.copy()
    new_delta_pos = delta_pos.copy()
    new_alpha_score = alpha_score
    new_beta_score = beta_score
    new_delta_score = delta_score

    for i in range(nPop):
        fitness = fitness_values[i]

        if fitness < new_alpha_score:
            new_delta_score = new_beta_score
            new_delta_pos = new_beta_pos.copy()
            new_beta_score = new_alpha_score
            new_beta_pos = new_alpha_pos.copy()
            new_alpha_score = fitness
            new_alpha_pos = positions[i].copy()
        elif fitness < new_beta_score:
            new_delta_score = new_beta_score
            new_delta_pos = new_beta_pos.copy()
            new_beta_score = fitness
            new_beta_pos = positions[i].copy()
        elif fitness < new_delta_score:
            new_delta_score = fitness
            new_delta_pos = positions[i].copy()

    return new_alpha_pos, new_beta_pos, new_delta_pos, new_alpha_score, new_beta_score, new_delta_score


class GreyWolfOptimizer:
    """Grey Wolf Optimizer.

    With no arguments, it optimizes the SAMA design problem using the settings
    in InData at the time it is constructed. Optional arguments (cost_function,
    var_min, var_max, n_pop, max_it, run_time, seed, generate_results, verbose)
    work as described for samapy.optimizers.swarm.Swarm.
    """
    def __init__(self, cost_function=None, var_min=None, var_max=None, n_pop=None,
                 max_it=None, run_time=None, seed=None, generate_results=True,
                 verbose=True, **kwargs):
        # Variable: PV number, WT number, Battery number, number of DG, Rated Power Inverter
        (self.VarMin, self.VarMax, self.nPop, self.MaxIt,
         self.Run_Time) = resolve_settings(InData, var_min, var_max, n_pop, max_it, run_time)
        self.nVar = len(self.VarMin)  # number of decision variables
        self.cost_function = cost_function
        self.seed = seed
        self.generate_results = generate_results
        self.verbose = verbose

        # Solution
        self.solution_best_costs = []
        self.solution_best_positions = []
        self.solution_cost_curve = []

    def _evaluate(self, positions):
        f = self.cost_function or cost_function
        return np.array([f(pos) for pos in positions])

    def optimize(self):
        """Run the optimization and return (best_position, best_cost)."""
        seed_all(self.seed)
        nPop, MaxIt = self.nPop, self.MaxIt
        if self.generate_results:
            plt.rcParams["font.family"] = "Times New Roman"
            fig, ax = plt.subplots()

        for tt in range(self.Run_Time):
            # Initialize the population of grey wolves
            positions = np.random.uniform(self.VarMin, self.VarMax, (nPop, self.nVar))

            # Initialize alpha, beta, and delta positions
            alpha_pos = np.zeros(self.nVar)
            alpha_score = float('inf')

            beta_pos = np.zeros(self.nVar)
            beta_score = float('inf')

            delta_pos = np.zeros(self.nVar)
            delta_score = float('inf')

            # Track Best Costs and Mean Costs
            best_cost, mean_cost = [], []

            # Initial fitness evaluation
            fitness_values = self._evaluate(positions)

            # Update initial leaders
            alpha_pos, beta_pos, delta_pos, alpha_score, beta_score, delta_score = \
                update_leaders_vectorized(fitness_values, positions, alpha_pos, beta_pos, delta_pos,
                                          alpha_score, beta_score, delta_score)

            # Main loop of GWO
            for it in range(MaxIt):
                # Calculate a (linearly decreases from 2 to 0)
                a = 2 - 2 * (it / MaxIt)

                # Update positions using vectorized JIT-compiled function
                positions = update_positions_vectorized(positions, alpha_pos, beta_pos, delta_pos,
                                                        a, self.VarMin, self.VarMax)

                # Evaluate fitness for all positions
                fitness_values = self._evaluate(positions)

                # Update leaders using vectorized JIT-compiled function
                alpha_pos, beta_pos, delta_pos, alpha_score, beta_score, delta_score = \
                    update_leaders_vectorized(fitness_values, positions, alpha_pos, beta_pos, delta_pos,
                                              alpha_score, beta_score, delta_score)

                # Calculate mean cost for this iteration
                current_mean_cost = np.mean(fitness_values)

                # Add new best cost and mean cost
                best_cost.append(alpha_score)
                mean_cost.append(current_mean_cost)

                # Print results for current iteration (reduce frequency for speed)
                if self.verbose and (it % 10 == 0 or it == MaxIt - 1):  # Print every 10 iterations
                    print(
                        f'Run time = {tt}, Iteration = {it + 1}, Best Cost = {round(alpha_score, 4)}, Mean Cost = {round(current_mean_cost, 4)}')

            self.solution_best_costs.append(alpha_score)
            self.solution_best_positions.append(alpha_pos.copy())
            self.solution_cost_curve.append(best_cost)

            if self.generate_results:
                ax.plot(best_cost, '-.', label=str(tt + 1))

        # Find the best solution across all runs
        Best = [self.solution_best_costs[t] for t in range(len(self.solution_best_positions))]
        index = np.argmin(Best)
        X = self.solution_best_positions[index]
        if not self.generate_results:
            return X, Best[index]

        # Generate results
        plt.xlabel('Iteration')
        plt.ylabel('Cost of Best Solution')
        plt.title('Grey Wolf Optimizer Convergence Curve')
        plt.legend()  # Display the legend
        plt.tight_layout()
        os.makedirs('samapy_outputs', exist_ok=True)
        plt.savefig('samapy_outputs/Optimization.png', dpi=300)
        print("✓ Saved: samapy_outputs/Optimization.png")
        Gen_Results(X)
        return X, Best[index]


# Alternative: Parallel implementation for multiple runs
class ParallelGreyWolfOptimizer(GreyWolfOptimizer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.n_processes = min(mp.cpu_count(), self.Run_Time)  # Use available cores

    def single_run(self, run_id):
        """Single GWO run for parallel execution"""
        # Different seed for each run (offset by self.seed when one is given)
        seed_all(run_id if self.seed is None else self.seed + run_id)
        nPop, MaxIt = self.nPop, self.MaxIt

        # Initialize the population of grey wolves
        positions = np.random.uniform(self.VarMin, self.VarMax, (nPop, self.nVar))

        # Initialize alpha, beta, and delta positions
        alpha_pos = np.zeros(self.nVar)
        alpha_score = float('inf')

        beta_pos = np.zeros(self.nVar)
        beta_score = float('inf')

        delta_pos = np.zeros(self.nVar)
        delta_score = float('inf')

        # Track Best Costs
        best_cost = []

        # Initial fitness evaluation
        fitness_values = self._evaluate(positions)

        # Update initial leaders
        alpha_pos, beta_pos, delta_pos, alpha_score, beta_score, delta_score = \
            update_leaders_vectorized(fitness_values, positions, alpha_pos, beta_pos, delta_pos,
                                      alpha_score, beta_score, delta_score)

        # Main loop of GWO
        for it in range(MaxIt):
            # Calculate a (linearly decreases from 2 to 0)
            a = 2 - 2 * (it / MaxIt)

            # Update positions using vectorized JIT-compiled function
            positions = update_positions_vectorized(positions, alpha_pos, beta_pos, delta_pos,
                                                    a, self.VarMin, self.VarMax)

            # Evaluate fitness for all positions
            fitness_values = self._evaluate(positions)

            # Update leaders using vectorized JIT-compiled function
            alpha_pos, beta_pos, delta_pos, alpha_score, beta_score, delta_score = \
                update_leaders_vectorized(fitness_values, positions, alpha_pos, beta_pos, delta_pos,
                                          alpha_score, beta_score, delta_score)

            # Add new best cost
            best_cost.append(alpha_score)

        return alpha_score, alpha_pos.copy(), best_cost

    def optimize_parallel(self):
        """Parallel optimization using multiprocessing"""
        plt.rcParams["font.family"] = "Times New Roman"
        fig, ax = plt.subplots()

        # Run multiple GWO instances in parallel
        with mp.Pool(processes=self.n_processes) as pool:
            results = pool.map(self.single_run, range(self.Run_Time))

        # Process results
        for tt, (best_cost_val, best_pos, cost_curve) in enumerate(results):
            self.solution_best_costs.append(best_cost_val)
            self.solution_best_positions.append(best_pos)
            self.solution_cost_curve.append(cost_curve)

            ax.plot(cost_curve, '-.', label=str(tt + 1))
            print(f'Run time = {tt}, Final Best Cost = {round(best_cost_val, 4)}')

        # Find the best solution across all runs
        Best = [self.solution_best_costs[t] for t in range(len(self.solution_best_positions))]
        index = np.argmin(Best)
        X = self.solution_best_positions[index]

        # Generate results
        plt.xlabel('Iteration')
        plt.ylabel('Cost of Best Solution')
        plt.title('Grey Wolf Optimizer Convergence Curve (Parallel)')
        plt.legend()  # Display the legend
        plt.tight_layout()
        import os
        os.makedirs('samapy_outputs', exist_ok=True)
        plt.savefig('samapy_outputs/Optimization.png', dpi=300)
        print("✓ Saved: samapy_outputs/Optimization.png")
