import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from Fitness import fitness as cost_function
from Results import Gen_Results
# Loading Inputs
from Input_Data import InData
import numpy as np


PV = InData.PV
WT = InData.WT
Bat = InData.Bat
DG = InData.DG
Run_Time = InData.Run_Time
nPop = InData.nPop
MaxIt = InData.MaxIt



class AdvancedDifferentialEvolution:
    def __init__(self, **kwargs):
        self.nVar = 5  # number of decision variables

        # Variable: PV number, WT number, Battery number, number of DG, Rated Power Inverter
        self.VarMin = np.array([0, 0, 0, 0, 0]) * [PV, WT, Bat, DG, 1]  # Lower bound of variables
        self.VarMax = np.array([60, 60, 60, 20, 60]) * [PV, WT, Bat, DG, 1]  # Upper bound of variables

        # DE parameters - adaptive and self-tuning
        self.F_min = 0.1  # Minimum scaling factor
        self.F_max = 0.9  # Maximum scaling factor
        self.CR_min = 0.1  # Minimum crossover probability
        self.CR_max = 0.9  # Maximum crossover probability

        # Advanced DE strategies
        self.strategies = ['DE/rand/1', 'DE/best/1', 'DE/current-to-best/1', 'DE/rand/2', 'DE/best/2']

        # Solution tracking
        self.solution_best_costs = []
        self.solution_best_positions = []
        self.solution_cost_curve = []

    def initialize_population(self):
        """Initialize population with enhanced diversity"""
        population = np.zeros((nPop, self.nVar))

        # Initialize with different strategies
        for i in range(nPop):
            if i < nPop // 4:
                # Random initialization
                population[i] = np.random.uniform(self.VarMin, self.VarMax)
            elif i < nPop // 2:
                # Center-biased initialization
                center = (self.VarMin + self.VarMax) / 2
                population[i] = center + np.random.normal(0, 0.25) * (self.VarMax - self.VarMin)
            elif i < 3 * nPop // 4:
                # Boundary-biased initialization
                for j in range(self.nVar):
                    if np.random.rand() < 0.5:
                        population[i, j] = self.VarMin[j] + 0.1 * (self.VarMax[j] - self.VarMin[j])
                    else:
                        population[i, j] = self.VarMax[j] - 0.1 * (self.VarMax[j] - self.VarMin[j])
            else:
                # Sobol-like initialization for better space coverage
                for j in range(self.nVar):
                    population[i, j] = self.VarMin[j] + ((i * 7 + j * 13) % 100) / 100 * (
                                self.VarMax[j] - self.VarMin[j])

        # Apply bounds
        population = np.clip(population, self.VarMin, self.VarMax)

        # Evaluate population
        costs = np.zeros(nPop)
        for i in range(nPop):
            try:
                costs[i] = cost_function(population[i])
                if not np.isfinite(costs[i]):
                    costs[i] = 1e10
            except Exception as e:
                print(f"Error evaluating initial population member {i}: {e}")
                costs[i] = 1e10

        return population, costs

    def adaptive_parameters(self, iteration, success_rate):
        """Adaptive parameter control based on success rate and iteration"""
        progress = iteration / MaxIt

        # Adaptive F (scaling factor)
        if success_rate > 0.2:
            F = self.F_min + (self.F_max - self.F_min) * (1 - progress)
        else:
            F = self.F_max * (1 - 0.5 * progress)

        # Adaptive CR (crossover probability)
        if success_rate > 0.15:
            CR = self.CR_min + (self.CR_max - self.CR_min) * progress
        else:
            CR = self.CR_max * (1 - 0.3 * progress)

        return F, CR

    def select_strategy(self, iteration, success_rates):
        """Select DE strategy based on recent performance"""
        if iteration < 10:
            return np.random.choice(self.strategies)
        else:
            # Weight strategies by their success rates
            weights = np.array([success_rates.get(strategy, 0.1) for strategy in self.strategies])
            weights = weights / np.sum(weights)
            return np.random.choice(self.strategies, p=weights)

    def mutation(self, population, best_idx, target_idx, F, strategy):
        """Advanced mutation with multiple strategies"""
        nPop, nVar = population.shape

        if strategy == 'DE/rand/1':
            # DE/rand/1: Vi = Xr1 + F * (Xr2 - Xr3)
            indices = np.random.choice(nPop, 3, replace=False)
            while target_idx in indices:
                indices = np.random.choice(nPop, 3, replace=False)
            mutant = population[indices[0]] + F * (population[indices[1]] - population[indices[2]])

        elif strategy == 'DE/best/1':
            # DE/best/1: Vi = Xbest + F * (Xr1 - Xr2)
            indices = np.random.choice(nPop, 2, replace=False)
            while target_idx in indices:
                indices = np.random.choice(nPop, 2, replace=False)
            mutant = population[best_idx] + F * (population[indices[0]] - population[indices[1]])

        elif strategy == 'DE/current-to-best/1':
            # DE/current-to-best/1: Vi = Xi + F * (Xbest - Xi) + F * (Xr1 - Xr2)
            indices = np.random.choice(nPop, 2, replace=False)
            while target_idx in indices:
                indices = np.random.choice(nPop, 2, replace=False)
            mutant = population[target_idx] + F * (population[best_idx] - population[target_idx]) + \
                     F * (population[indices[0]] - population[indices[1]])

        elif strategy == 'DE/rand/2':
            # DE/rand/2: Vi = Xr1 + F * (Xr2 - Xr3) + F * (Xr4 - Xr5)
            indices = np.random.choice(nPop, 5, replace=False)
            while target_idx in indices:
                indices = np.random.choice(nPop, 5, replace=False)
            mutant = population[indices[0]] + F * (population[indices[1]] - population[indices[2]]) + \
                     F * (population[indices[3]] - population[indices[4]])

        elif strategy == 'DE/best/2':
            # DE/best/2: Vi = Xbest + F * (Xr1 - Xr2) + F * (Xr3 - Xr4)
            indices = np.random.choice(nPop, 4, replace=False)
            while target_idx in indices:
                indices = np.random.choice(nPop, 4, replace=False)
            mutant = population[best_idx] + F * (population[indices[0]] - population[indices[1]]) + \
                     F * (population[indices[2]] - population[indices[3]])

        # Apply bounds
        mutant = np.clip(mutant, self.VarMin, self.VarMax)
        return mutant

    def crossover(self, target, mutant, CR):
        """Binomial crossover with enhancements"""
        nVar = len(target)
        trial = target.copy()

        # Ensure at least one parameter is from mutant
        j_rand = np.random.randint(0, nVar)

        for j in range(nVar):
            if np.random.rand() < CR or j == j_rand:
                trial[j] = mutant[j]

        return trial

    def local_search(self, individual, cost, iteration):
        """Optional local search for promising solutions"""
        if iteration % 20 == 0 and cost < np.median([1e10]):  # Only for good solutions
            best_individual = individual.copy()
            best_cost = cost

            # Simple local search around current solution
            for _ in range(5):
                perturbed = individual + np.random.normal(0, 0.05) * (self.VarMax - self.VarMin)
                perturbed = np.clip(perturbed, self.VarMin, self.VarMax)

                try:
                    perturbed_cost = cost_function(perturbed)
                    if np.isfinite(perturbed_cost) and perturbed_cost < best_cost:
                        best_individual = perturbed
                        best_cost = perturbed_cost
                except:
                    continue

            return best_individual, best_cost

        return individual, cost

    def optimize(self):
        plt.rcParams["font.family"] = "Times New Roman"
        fig, ax = plt.subplots()

        for tt in range(Run_Time):
            # Initialize population
            population, costs = self.initialize_population()

            # Track best solution
            best_idx = np.argmin(costs)
            best_individual = population[best_idx].copy()
            best_cost = costs[best_idx]

            # Track performance
            best_cost_history = []
            mean_cost_history = []
            strategy_success = {strategy: 0 for strategy in self.strategies}
            strategy_attempts = {strategy: 0 for strategy in self.strategies}

            successful_mutations = 0
            total_mutations = 0

            # Main DE loop
            for iteration in range(1, MaxIt + 1):
                # Calculate success rate
                success_rate = successful_mutations / max(total_mutations, 1)

                # Adaptive parameters
                F, CR = self.adaptive_parameters(iteration, success_rate)

                # Calculate strategy success rates
                success_rates = {}
                for strategy in self.strategies:
                    if strategy_attempts[strategy] > 0:
                        success_rates[strategy] = strategy_success[strategy] / strategy_attempts[strategy]
                    else:
                        success_rates[strategy] = 0.1

                new_population = population.copy()
                improvements = 0

                for i in range(nPop):
                    # Select strategy
                    strategy = self.select_strategy(iteration, success_rates)
                    strategy_attempts[strategy] += 1

                    # Mutation
                    mutant = self.mutation(population, best_idx, i, F, strategy)

                    # Crossover
                    trial = self.crossover(population[i], mutant, CR)

                    # Evaluation
                    try:
                        trial_cost = cost_function(trial)
                        if not np.isfinite(trial_cost):
                            trial_cost = 1e10
                    except:
                        trial_cost = 1e10

                    total_mutations += 1

                    # Selection
                    if trial_cost < costs[i]:
                        new_population[i] = trial
                        costs[i] = trial_cost
                        successful_mutations += 1
                        improvements += 1
                        strategy_success[strategy] += 1

                        # Update global best
                        if trial_cost < best_cost:
                            best_individual = trial.copy()
                            best_cost = trial_cost
                            best_idx = i

                population = new_population

                # Optional local search for best solutions
                if iteration % 30 == 0:
                    best_individual, best_cost = self.local_search(best_individual, best_cost, iteration)

                # Record progress
                best_cost_history.append(best_cost)
                mean_cost_history.append(np.mean(costs))

                # Print progress
                if iteration % 10 == 0 or iteration == 1:
                    print(f'Run {tt}, Iteration {iteration}, Best Cost = {best_cost:.4f}, '
                          f'Mean Cost = {mean_cost_history[-1]:.4f}, Success Rate = {success_rate:.3f}')

                # Diversity maintenance
                if iteration % 50 == 0:
                    diversity = np.mean(np.std(population, axis=0))
                    if diversity < 0.01 * np.mean(self.VarMax - self.VarMin):
                        # Reinitialize worst 10% of population
                        worst_indices = np.argsort(costs)[-nPop // 10:]
                        for idx in worst_indices:
                            population[idx] = np.random.uniform(self.VarMin, self.VarMax)
                            try:
                                costs[idx] = cost_function(population[idx])
                                if not np.isfinite(costs[idx]):
                                    costs[idx] = 1e10
                            except:
                                costs[idx] = 1e10

            # Store results
            self.solution_best_costs.append(best_cost)
            self.solution_best_positions.append(best_individual)
            self.solution_cost_curve.append(best_cost_history)

            # Plot convergence
            ax.plot(best_cost_history, '-.', label=f'Run {tt + 1}')

        # Find overall best solution
        best_run_idx = np.argmin(self.solution_best_costs)
        best_solution = self.solution_best_positions[best_run_idx]

        # Generate plots
        plt.xlabel('Iteration')
        plt.ylabel('Cost of Best Solution')
        plt.title('Advanced Differential Evolution - Convergence Curve')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig('output/figs/DE_Optimization.png', dpi=300)

        # Generate results
        Gen_Results(best_solution)