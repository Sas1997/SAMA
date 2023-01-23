from copy import copy, deepcopy
import numpy as np
from numba import njit

from input_data import *
from fitness import fitness as cost_function

class Swarm:
    def __init__(self, **kwargs):
        self.nVar = 5                          # number of decision variables

        # Variable: PV number, WT number, Battery number, number of DG, Rated Power Inverter
        self.VarMin = np.array([0,0,0,0,0]) * [PV,WT,Bat,DG,1] # Lower bound of variables
        self.VarMax = np.array([100,100,60,10,20]) * [PV,WT,Bat,DG,1] # Upper bound of variables

        # Velocity limits
        self.VelMax = 0.3 * (self.VarMax - self.VarMin)
        self.VelMin = -self.VelMax

        # %% PSO Parameters
        self.run_time = 1
        self.MaxIt = 100      # Max number of iterations
        self.nPop = 50        # Population size (swarm size)
        self.w = 1            # Inertia weight
        self.wdamp = 0.99     # Inertia weight damping ratio
        self.c1 = 2           # Personal learning coefficient
        self.c2 = 2           # Global learning coefficient

        # Solution
        self.solution = []

    def optimize(self):

        for tt in range(self.run_time):

            # Initialize particle positions
            particle_positions = np.random.uniform(self.VarMin, self.VarMax, (1, self.nPop, self.nVar))[0]
            particle_personal_best_position = deepcopy(particle_positions)

            # Initialize particle velocities
            particle_velocities = np.zeros((self.nPop, self.nVar))

            # Evaluate costs per initial particle
            particle_costs = np.apply_along_axis(cost_function, 1, particle_positions)
            particle_personal_best_cost = deepcopy(particle_costs)

            # Determine global best
            min_cost_index = np.argmin(particle_personal_best_cost)
            global_best_position = deepcopy(particle_personal_best_position[min_cost_index])
            global_best_cost = particle_personal_best_cost[min_cost_index]

            # Track Best Costs and Mean Costs
            best_cost, mean_cost = [], []

            # PSO Main Loop
            for it in range(self.MaxIt):
                for i in range(self.nPop):
                    # Update Velocity
                    particle_velocities[i] = self.w*particle_velocities[i]+self.c1*np.random.rand(self.nVar)\
                        *(particle_personal_best_position[i]-particle_positions[i])\
                            +self.c2*np.random.rand(self.nVar)*(global_best_position-particle_positions[i])
                    
                    # Apply Velocity Limits
                    particle_velocities[i] = np.minimum(np.maximum(particle_velocities[i],self.VelMin),self.VelMax)
                    
                    # Update Position
                    particle_positions[i] += particle_velocities[i]
                    
                    # Velocity Mirror Effect
                    is_outside=(np.less(particle_positions[i], self.VarMin) | np.greater(particle_positions[i], self.VarMax))[0]
                    particle_velocities[i][is_outside]=-particle_velocities[i][is_outside]
                    
                    # Apply Position Limits
                    particle_positions[i] = np.minimum(np.maximum(particle_positions[i], self.VarMin), self.VarMax)

                    # Evaluation
                    particle_costs[i] = cost_function(particle_positions[i])

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
                mean_cost.append(sum(particle_personal_best_cost) / self.nPop)

                # Update inertia factor
                self.w *= self.wdamp

                # Print results for current iteration
                print(f'Run time = {tt}, Iteration = {it}, Best Cost = {round(global_best_cost, 4)}, Mean Cost = {round(mean_cost[-1], 4)}')

            self.solution.append({
                "cost": global_best_cost,
                "position": global_best_position,
                "cost_curve": best_cost
            })