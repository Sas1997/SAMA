from copy import copy, deepcopy
import numpy as np
# from numba.experimental import jitclass
import matplotlib.pyplot as plt

from input_data import *
from fitness import fitness as cost_function

# @jitclass # use all keywords
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
        self.solution_best_costs = []
        self.solution_best_positions = []
        self.solution_cost_curve = []

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
            for it in range(1, self.MaxIt+1):
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

            self.solution_best_costs.append(global_best_cost)
            self.solution_best_positions.append(global_best_position)
            self.solution_cost_curve.append(best_cost)
            
    def get_final_result(self, print_result=False, plot_curve=False):
        index = np.argmin(np.array(self.solution_best_costs))
        
        X = self.solution_best_positions[index]
        Cash_Flow, Pbuy, Psell, Eload, Ens, Pdg, Pch, Pdch, Ppv, Pwt, Eb, Cn_B, Edump = cost_function(X, final_solution=True, print_result=print_result)

        if plot_curve:
            # Result 1
            plt.figure()
            plt.plot(self.solution_cost_curve[index], '-.')
            plt.xlabel('iteration')
            plt.ylabel('Cost of Best Solution ')
            plt.title('Converage Curve')

            # Result
            plt.figure()
            for kk in range(5):
                plt.bar(range(0,25),Cash_Flow[:,kk])
            plt.legend(['Capital','Operating','Salvage','Fuel','Replacement'])
            plt.title('Cash Flow')
            plt.xlabel('Year')
            plt.ylabel('$')

            # Plot Results
            plt.figure()
            plt.plot(Pbuy)
            plt.plot(Psell)
            plt.legend(['Buy','sell'])
            plt.ylabel('Pgrid (kWh)')
            plt.xlabel('t(hour)')

            plt.figure()
            plt.plot(Eload-Ens,'b-.')
            plt.plot(Pdg,'r')
            plt.plot(Pch-Pdch,'g')
            plt.plot(Ppv+Pwt,'--')
            plt.legend(['Load-Ens','Pdg','Pbat','P_{RE}'])

            plt.figure()
            plt.plot(Eb/Cn_B)
            plt.title('State of Charge')
            plt.ylabel('SOC')
            plt.xlabel('t[hour]')

            # Plot results for one specific day 
            Day=180;
            t1=Day*24+1;
            t2=Day*24+24;

            plt.figure(figsize=(10,10))
            plt.title(['Results for ' ,str(Day), ' -th day']) 
            plt.subplot(4,4,1)
            plt.plot(Eload)
            plt.title('Load Profile')
            plt.ylabel('E_{load} [kWh]')
            plt.xlabel('t[hour]')
            plt.xlim([t1,t2])

            plt.subplot(4,4,5)
            plt.plot(Eload)
            plt.title('Load Profile')
            plt.ylabel('E_{load} [kWh]')
            plt.xlabel('t[hour]')
            plt.xlim([t1,t2])

            plt.subplot(4,4,2)
            plt.plot(G)
            plt.title('Plane of Array Irradiance')
            plt.ylabel('G[W/m^2]')
            plt.xlabel('t[hour]')
            plt.xlim([t1, t2])

            plt.subplot(4,4,6)
            plt.plot(T)
            plt.title('Ambient Temperature')
            plt.ylabel('T[^o C]')
            plt.xlabel('t[hour]')
            plt.xlim([t1 ,t2])

            plt.subplot(4,4,3)
            plt.plot(Ppv)
            plt.title('PV Power')
            plt.ylabel('P_{pv} [kWh]')
            plt.xlabel('t[hour]')
            plt.xlim([t1, t2])

            plt.subplot(4,4,4)
            plt.plot(Ppv)
            plt.title('PV Power')
            plt.ylabel('P_{pv} [kWh]')
            plt.xlabel('t[hour]')
            plt.xlim([t1, t2])

            plt.subplot(4,4,7)
            plt.plot(Pwt)
            plt.title('WT Energy')
            plt.ylabel('P_{wt} [kWh]')
            plt.xlabel('t[hour]')
            plt.xlim([t1, t2])
            plt.subplot(4,4,8)
            plt.plot(Pwt)
            plt.title('WT Energy')
            plt.ylabel('P_{wt} [kWh]')
            plt.xlabel('t[hour]')
            plt.xlim([t1, t2])

            plt.subplot(4,4,9)
            plt.plot(Pdg)
            plt.title('Diesel Generator Energy')
            plt.ylabel('E_{DG} [kWh]')
            plt.xlabel('t[hour]')
            plt.xlim([t1,t2])
            plt.subplot(4,4,10)
            plt.plot(Pdg)
            plt.title('Diesel Generator Energy')
            plt.ylabel('E_{DG} [kWh]')
            plt.xlabel('t[hour]')
            plt.xlim([t1,t2])

            plt.subplot(4,4,11)
            plt.plot(Eb)
            plt.title('Battery Energy Level')
            plt.ylabel('E_{b} [kWh]')
            plt.xlabel('t[hour]')
            plt.xlim([t1, t2])

            plt.subplot(4,4,12)
            plt.plot(Eb/Cn_B)
            plt.title('State of Charge')
            plt.ylabel('SOC')
            plt.xlabel('t[hour]')
            plt.xlim([t1,t2])

            plt.subplot(4,4,13)
            plt.plot(Ens)
            plt.title('Loss of Power Suply')
            plt.ylabel('LPS[kWh]')
            plt.xlabel('t[hour]')
            plt.xlim([t1,t2])

            plt.subplot(4,4,14)
            plt.plot(Edump)
            plt.title('Dumped Energy')
            plt.ylabel('E_{dump} [kWh]')
            plt.xlabel('t[hour]')
            plt.xlim([t1,t2])

            plt.subplot(4,4,15)
            plt.bar(range(len(Pdch)),Pdch)
            plt.title('Battery decharge Energy')
            plt.ylabel('E_{dch} [kWh]')
            plt.xlabel('t[hour]')
            plt.xlim([t1, t2])

            plt.subplot(4,4,16)
            plt.bar(range(len(Pdch)),Pch)
            plt.title('Battery charge Energy')
            plt.ylabel('E_{ch} [kWh]')
            plt.xlabel('t[hour]')
            plt.xlim([t1,t2])

            plt.show()


