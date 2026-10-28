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
from copy import copy, deepcopy
from time import process_time

start = process_time()

PV = InData.PV
WT = InData.WT
Bat = InData.Bat
DG = InData.DG
Run_Time = InData.Run_Time
nPop = InData.nPop
MaxIt = InData.MaxIt
c1 = InData.c1
c2 = InData.c2
wdamp = InData.wdamp
VarMin = InData.VarMin
VarMax = InData.VarMax

# Problem Definition
from samapy.core.Fitness import fitness as cost_function
from samapy.optimizers._common import seed_all, resolve_settings


class Swarm:
    """Particle Swarm Optimization.

    With no arguments, Swarm() optimizes the SAMA design problem using the
    settings in InData at the time it is constructed. The optional arguments
    let the same algorithm run on another objective (for example a benchmark
    function in the test suite):

        cost_function     callable taking a 1-D array, returning a float
        var_min, var_max  lower and upper bounds (sequences of equal length)
        n_pop, max_it, run_time  population size, iterations, independent runs
        seed              integer seed for reproducible runs
        generate_results  False skips the SAMA convergence plot and Gen_Results
        verbose           False suppresses per-iteration printing
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
        self.w = InData.w
        self.c1 = InData.c1
        self.c2 = InData.c2
        self.wdamp = InData.wdamp

        # Velocity limits
        self.VelMax = 0.3 * (self.VarMax - self.VarMin)
        self.VelMin = -self.VelMax
        # Solution
        self.solution_best_costs = []
        self.solution_best_positions = []
        self.solution_cost_curve = []

    def _cost(self, x):
        return (self.cost_function or cost_function)(x)

    def optimize(self):
        """Run the optimization and return (best_position, best_cost)."""
        seed_all(self.seed)
        nPop, MaxIt = self.nPop, self.MaxIt
        c1, c2, wdamp = self.c1, self.c2, self.wdamp
        w = self.w
        if self.generate_results:
            plt.rcParams["font.family"] = "Times New Roman"
            fig, ax = plt.subplots()
        for tt in range(self.Run_Time):

            # Initialize particle positions
            particle_positions = np.random.uniform(self.VarMin, self.VarMax, (1, nPop, self.nVar))[0]
            particle_personal_best_position = deepcopy(particle_positions)

            # Initialize particle velocities
            particle_velocities = np.zeros((nPop, self.nVar))

            # Evaluate costs per initial particle
            particle_costs = np.apply_along_axis(self._cost, 1, particle_positions)
            particle_personal_best_cost = deepcopy(particle_costs)

            # Determine global best
            min_cost_index = np.argmin(particle_personal_best_cost)
            global_best_position = deepcopy(particle_personal_best_position[min_cost_index])
            global_best_cost = particle_personal_best_cost[min_cost_index]

            # Track Best Costs and Mean Costs
            best_cost, mean_cost = [], []

            # PSO Main Loop
            for it in range(1, MaxIt + 1):
                for i in range(nPop):
                    # Update Velocity
                    particle_velocities[i] = w * particle_velocities[i] + c1 * np.random.rand(self.nVar) \
                                             * (particle_personal_best_position[i] - particle_positions[i]) \
                                             + c2 * np.random.rand(self.nVar) * (
                                                         global_best_position - particle_positions[i])

                    # Apply Velocity Limits
                    particle_velocities[i] = np.minimum(np.maximum(particle_velocities[i], self.VelMin), self.VelMax)

                    # Update Position
                    particle_positions[i] += particle_velocities[i]

                    # Velocity Mirror Effect
                    is_outside = \
                    (np.less(particle_positions[i], self.VarMin) | np.greater(particle_positions[i], self.VarMax))[0]
                    particle_velocities[i][is_outside] = -particle_velocities[i][is_outside]

                    # Apply Position Limits
                    particle_positions[i] = np.minimum(np.maximum(particle_positions[i], self.VarMin), self.VarMax)

                    # Evaluation
                    particle_costs[i] = self._cost(particle_positions[i])

                    # Update Personal Best
                    if particle_costs[i] < particle_personal_best_cost[i]:
                        particle_personal_best_position[i] = particle_positions[i]
                        particle_personal_best_cost[i] = particle_costs[i]

                        # Update Global Best
                        if particle_personal_best_cost[i] < global_best_cost:
                            global_best_cost = particle_personal_best_cost[i]
                            global_best_position = particle_personal_best_position[i]

                # Add new best cost and mean cost
                best_cost.append(global_best_cost)
                mean_cost.append(sum(particle_personal_best_cost) / nPop)

                # Update inertia factor
                w *= wdamp

                # Print results for current iteration
                if self.verbose:
                    print(
                        f'Run time = {tt}, Iteration = {it}, Best Cost = {round(global_best_cost, 4)}, Mean Cost = {round(mean_cost[-1], 4)}')

            self.solution_best_costs.append(global_best_cost)
            self.solution_best_positions.append(global_best_position)
            self.solution_cost_curve.append(best_cost)

            if self.generate_results:
                ax.plot(best_cost, '-.', label=str(tt+1))

        Best= [self.solution_best_costs[t] for t in range(len(self.solution_best_positions))]
        index = np.argmin(Best)
        X = self.solution_best_positions[index]
        if not self.generate_results:
            return X, Best[index]

        # Run Results file

        plt.xlabel('Iteration')
        plt.ylabel('Cost of Best Solution')
        plt.title('Convergence curve')
        plt.legend()  # Display the legend
        plt.tight_layout()
        import os
        os.makedirs('samapy_outputs', exist_ok=True)
        plt.savefig('samapy_outputs/Optimization.png', dpi=300)
        print("✓ Saved: samapy_outputs/Optimization.png")
        Gen_Results(X)
        return X, Best[index]
