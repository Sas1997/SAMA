import matplotlib

matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from Results import Gen_Results
# Loading Inputs
from Input_Data import InData
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

# Problem Definition
from Fitness import fitness as cost_function


class ImprovedArtificialBeeColony:
    def __init__(self, **kwargs):
        self.nVar = 5  # number of decision variables

        # Variable: PV number, WT number, Battery number, number of DG, Rated Power Inverter
        self.VarMin = np.array([0, 0, 0, 0, 0]) * [PV, WT, Bat, DG, 1]  # Lower bound of variables
        self.VarMax = np.array([60, 60, 60, 20, 60]) * [PV, WT, Bat, DG, 1]  # Upper bound of variables

        # Improved ABC parameters
        self.nEmployedBees = nPop // 2  # Number of employed bees
        self.nOnlookerBees = nPop // 2  # Number of onlooker bees
        self.maxTrials = 15  # Increased for better exploration
        self.modification_rate = 0.8  # Probability of modifying multiple dimensions

        # Adaptive parameters
        self.initial_search_radius = 0.5
        self.final_search_radius = 0.1

        # Solution
        self.solution_best_costs = []
        self.solution_best_positions = []
        self.solution_cost_curve = []

    def initialize_food_sources(self):
        """Initialize food sources using improved initialization"""
        positions = np.random.uniform(self.VarMin, self.VarMax, (self.nEmployedBees, self.nVar))

        # Add some positions with good initial guesses
        for i in range(min(5, self.nEmployedBees)):
            # Initialize with reasonable starting points
            positions[i] = (self.VarMin + self.VarMax) / 2 + np.random.normal(0, 0.1) * (self.VarMax - self.VarMin)
            positions[i] = np.minimum(np.maximum(positions[i], self.VarMin), self.VarMax)

        costs = np.apply_along_axis(cost_function, 1, positions)
        trials = np.zeros(self.nEmployedBees)
        return positions, costs, trials

    def adaptive_search_radius(self, iteration):
        """Calculate adaptive search radius based on iteration"""
        progress = iteration / MaxIt
        return self.initial_search_radius * (1 - progress) + self.final_search_radius * progress

    def improved_employed_bees_phase(self, positions, costs, trials, iteration, global_best_position):
        """Improved employed bees phase with multiple search strategies"""
        new_positions = positions.copy()
        new_costs = costs.copy()
        search_radius = self.adaptive_search_radius(iteration)

        for i in range(self.nEmployedBees):
            # Strategy 1: Multi-dimensional modification (80% chance)
            if np.random.rand() < self.modification_rate:
                # Modify multiple dimensions
                n_dims = np.random.randint(1, self.nVar + 1)
                dims_to_modify = np.random.choice(self.nVar, n_dims, replace=False)

                new_position = positions[i].copy()

                for j in dims_to_modify:
                    # Choose search strategy randomly
                    strategy = np.random.choice(['neighbor', 'global_best', 'random_walk'])

                    if strategy == 'neighbor':
                        # Select random neighbor
                        k = np.random.randint(0, self.nEmployedBees)
                        while k == i:
                            k = np.random.randint(0, self.nEmployedBees)
                        phi = np.random.uniform(-search_radius, search_radius)
                        new_position[j] = positions[i][j] + phi * (positions[i][j] - positions[k][j])

                    elif strategy == 'global_best':
                        # Move towards global best
                        phi = np.random.uniform(-search_radius, search_radius)
                        new_position[j] = positions[i][j] + phi * (global_best_position[j] - positions[i][j])

                    else:  # random_walk
                        # Random walk
                        step_size = search_radius * (self.VarMax[j] - self.VarMin[j])
                        new_position[j] = positions[i][j] + np.random.normal(0, step_size)

            else:
                # Strategy 2: Single dimension modification with larger step
                j = np.random.randint(0, self.nVar)
                k = np.random.randint(0, self.nEmployedBees)
                while k == i:
                    k = np.random.randint(0, self.nEmployedBees)

                new_position = positions[i].copy()
                phi = np.random.uniform(-2 * search_radius, 2 * search_radius)
                new_position[j] = positions[i][j] + phi * (positions[i][j] - positions[k][j])

            # Apply bounds
            new_position = np.minimum(np.maximum(new_position, self.VarMin), self.VarMax)

            # Evaluate new solution
            new_cost = cost_function(new_position)

            # Greedy selection with elitism
            if new_cost < costs[i]:
                new_positions[i] = new_position
                new_costs[i] = new_cost
                trials[i] = 0
            else:
                trials[i] += 1

        return new_positions, new_costs, trials

    def calculate_probabilities(self, costs):
        """Improved probability calculation with better selection pressure"""
        # Convert costs to fitness values with exponential scaling
        min_cost = np.min(costs)
        max_cost = np.max(costs)

        if max_cost == min_cost:
            probabilities = np.ones(len(costs)) / len(costs)
        else:
            # Normalize and apply exponential scaling
            normalized_costs = (costs - min_cost) / (max_cost - min_cost)
            fitness = np.exp(-3 * normalized_costs)  # Exponential scaling for better selection pressure
            probabilities = fitness / np.sum(fitness)

        return probabilities

    def improved_onlooker_bees_phase(self, positions, costs, trials, iteration, global_best_position):
        """Improved onlooker bees phase with better exploitation"""
        probabilities = self.calculate_probabilities(costs)
        new_positions = positions.copy()
        new_costs = costs.copy()
        search_radius = self.adaptive_search_radius(iteration)

        for _ in range(self.nOnlookerBees):
            # Select food source based on probability (with tournament selection for better diversity)
            if np.random.rand() < 0.7:  # 70% probability-based selection
                i = np.random.choice(self.nEmployedBees, p=probabilities)
            else:  # 30% tournament selection
                tournament_size = 3
                candidates = np.random.choice(self.nEmployedBees, tournament_size, replace=False)
                i = candidates[np.argmin(costs[candidates])]

            # Improved search around selected food source
            if np.random.rand() < 0.6:  # 60% chance for global best guided search
                # Move towards global best with exploration
                new_position = positions[i].copy()
                for j in range(self.nVar):
                    if np.random.rand() < 0.5:  # 50% chance to modify each dimension
                        phi = np.random.uniform(-search_radius, search_radius)
                        new_position[j] = positions[i][j] + phi * (global_best_position[j] - positions[i][j])
                        # Add small random perturbation
                        new_position[j] += np.random.normal(0, 0.05 * (self.VarMax[j] - self.VarMin[j]))
            else:
                # Regular neighbor-based search
                j = np.random.randint(0, self.nVar)
                k = np.random.randint(0, self.nEmployedBees)
                while k == i:
                    k = np.random.randint(0, self.nEmployedBees)

                new_position = positions[i].copy()
                phi = np.random.uniform(-search_radius, search_radius)
                new_position[j] = positions[i][j] + phi * (positions[i][j] - positions[k][j])

            # Apply bounds
            new_position = np.minimum(np.maximum(new_position, self.VarMin), self.VarMax)

            # Evaluate new solution
            new_cost = cost_function(new_position)

            # Greedy selection
            if new_cost < costs[i]:
                new_positions[i] = new_position
                new_costs[i] = new_cost
                trials[i] = 0
            else:
                trials[i] += 1

        return new_positions, new_costs, trials

    def improved_scout_bees_phase(self, positions, costs, trials, global_best_position):
        """Improved scout bees phase with guided reinitialization"""
        new_positions = positions.copy()
        new_costs = costs.copy()

        for i in range(self.nEmployedBees):
            if trials[i] > self.maxTrials:
                # 50% chance for guided reinitialization, 50% for random
                if np.random.rand() < 0.5:
                    # Guided reinitialization around global best
                    new_positions[i] = global_best_position + np.random.normal(0, 0.3, self.nVar) * (
                                self.VarMax - self.VarMin)
                else:
                    # Random reinitialization
                    new_positions[i] = np.random.uniform(self.VarMin, self.VarMax)

                # Apply bounds
                new_positions[i] = np.minimum(np.maximum(new_positions[i], self.VarMin), self.VarMax)
                new_costs[i] = cost_function(new_positions[i])
                trials[i] = 0

        return new_positions, new_costs, trials

    def optimize(self):
        plt.rcParams["font.family"] = "Times New Roman"
        fig, ax = plt.subplots()

        for tt in range(Run_Time):
            # Initialize food sources
            positions, costs, trials = self.initialize_food_sources()

            # Find initial best solution
            best_index = np.argmin(costs)
            global_best_position = positions[best_index].copy()
            global_best_cost = costs[best_index]

            # Track Best Costs and Mean Costs
            best_cost, mean_cost = [], []

            # Variables for convergence acceleration
            stagnation_counter = 0
            last_improvement = global_best_cost

            # ABC Main Loop
            for it in range(1, MaxIt + 1):
                # Improved employed bees phase
                positions, costs, trials = self.improved_employed_bees_phase(
                    positions, costs, trials, it, global_best_position)

                # Improved onlooker bees phase
                positions, costs, trials = self.improved_onlooker_bees_phase(
                    positions, costs, trials, it, global_best_position)

                # Update global best
                current_best_index = np.argmin(costs)
                if costs[current_best_index] < global_best_cost:
                    improvement = global_best_cost - costs[current_best_index]
                    global_best_cost = costs[current_best_index]
                    global_best_position = positions[current_best_index].copy()
                    stagnation_counter = 0
                else:
                    stagnation_counter += 1

                # Improved scout bees phase
                positions, costs, trials = self.improved_scout_bees_phase(
                    positions, costs, trials, global_best_position)

                # Convergence acceleration: if stagnated for too long, diversify
                if stagnation_counter > 20:
                    # Reinitialize worst 20% of population
                    worst_indices = np.argsort(costs)[-self.nEmployedBees // 5:]
                    for idx in worst_indices:
                        positions[idx] = np.random.uniform(self.VarMin, self.VarMax)
                        costs[idx] = cost_function(positions[idx])
                        trials[idx] = 0
                    stagnation_counter = 0

                # Add new best cost and mean cost
                best_cost.append(global_best_cost)
                mean_cost.append(np.mean(costs))

                # Print results for current iteration
                if it % 10 == 0 or it == 1:  # Print every 10 iterations to reduce output
                    print(
                        f'Run time = {tt}, Iteration = {it}, Best Cost = {round(global_best_cost, 4)}, Mean Cost = {round(mean_cost[-1], 4)}')

            self.solution_best_costs.append(global_best_cost)
            self.solution_best_positions.append(global_best_position)
            self.solution_cost_curve.append(best_cost)

            ax.plot(best_cost, '-.', label=str(tt + 1))

        # Find overall best solution
        Best = [self.solution_best_costs[t] for t in range(len(self.solution_best_positions))]
        index = np.argmin(Best)
        X = self.solution_best_positions[index]

        # Plot results
        plt.xlabel('Iteration')
        plt.ylabel('Cost of Best Solution')
        plt.title('Improved ABC Optimization Convergence Curve')
        plt.legend()
        plt.tight_layout()
        plt.savefig('output/figs/Improved_ABC_Optimization.png', dpi=300)

        # Run Results file
        Gen_Results(X)

        print(f"Best solution found: {X}")
        print(f"Best cost: {self.solution_best_costs[index]}")