import numpy as np
import pandas as pd

from Input_Data import Data
from EMS import energy_management
from Battery_Model import battery_model
from Fitness import fitness
from Models import Solution, Particle

# https://mathesaurus.sourceforge.net/matlab-numpy.html



"""
Main method
"""
def main(data='Data.csv', **kwargs):
    # read data from csv
    df = pd.read_csv(data, header=None)
    Eload = df[0]
    G = df[1]
    T = df[2]
    Vw = df[3]

    inputs = Data(Eload, G, T, Vw) # load input data
    inputs.set_user_data(**kwargs) # set user defined values

    # run PSO and when we get the best cost's position, we put them into the fitness function to get those results
    position = pso(Eload, G, T, Vw, inputs)
    
    print(position)
    
    # LCOE = fitness(position, Eload, G, T, Vw, inputs) 

    # print(LCOE)


"""
PSO function
"""
def pso(
    Eload, 
    G, 
    T, 
    Vw,
    inputs
):

    nVar = 5                # number of decision variables
    VarSize = (1, nVar)   # size of decision variables matrix

    # Variable: PV number, WT number, Battery number, number of DG, Rated Power Inverter
    VarMin = np.array([0,0,0,0,0]) # Lower bound of variables
    VarMax = np.array([100,100,60,10,20]) # Upper bound of variables

    VarMin = VarMin * [inputs.PV, inputs.WT, inputs.Bat, inputs.DG, 1]
    VarMax = VarMax * [inputs.PV, inputs.WT, inputs.Bat, inputs.DG, 1]

    # PSO parameters
    MaxIt = 100 # Max number of iterations
    nPop = 50 # Population size (swarm size)
    w = 1 # Inertia weight
    wdamp = 0.99 # Inertia weight damping ratio
    c1 = 2 # Personal learning coefficient
    c2 = 2 # Global learning coefficient

    # Velocity limits
    VelMax = 0.3 * (VarMax - VarMin)
    VelMin = -VelMax

    Run_Time = 1

    solution_particle = Solution()

    FinalBest = {
        "Cost": float('inf'),
        "Position": None
    }

    for tt in range(Run_Time):
        w = 1 # intertia weight 

        # initialization
        empty_particle = Particle()
        particle = [empty_particle for _ in range(nPop)]
        particle = np.array(particle)

        GlobalBest = {
            "Cost": float('inf'),
            "Position": None
        }

        for i in range(nPop):
            # initialize position
            position_array = []
            for var in range(len(VarMin)):
                position_array.append(np.random.uniform(VarMin[var], VarMax[var]))
            particle[i].Position = np.array(position_array)
            
            # initialize velocity
            particle[i].Velocity = np.zeros(VarSize)
            
            # evaluation
            particle[i].Cost = fitness(particle[i].Position, Eload, G, T, Vw, inputs)
            
            # update personal best
            particle[i].BestPosition = particle[i].Position
            particle[i].BestCost = particle[i].Cost

            # Update global best
            if particle[i].BestCost < GlobalBest["Cost"]:
                GlobalBest["Cost"] = particle[i].BestCost
                GlobalBest["Position"] = particle[i].BestPosition
    
        BestCost = np.zeros((MaxIt, 1))
        MeanCost = np.zeros((MaxIt, 1))

        # PSO main loop
        for it in range(MaxIt):
            for i in range(nPop):

                # update velocity
                particle[i].Velocity = w * particle[i].Velocity + c1 * np.random.uniform(0,1,(VarSize)) * (particle[i].BestPosition - particle[i].Position) + c2 * np.random.uniform(0,1,(VarSize)) * (GlobalBest["Position"]-particle[i].Position)

                # apply velocity limits
                particle[i].Velocity = np.maximum(particle[i].Velocity, VelMin)
                particle[i].Velocity = np.minimum(particle[i].Velocity, VelMax)

                # update position
                particle[i].Position = particle[i].Position + particle[i].Velocity

                # Velocity Mirror Effect
                # TODO: double check this condition is correct
                if np.any(np.less(particle[i].Position, VarMin) | np.greater(particle[i].Position, VarMax)):
                    particle[i].Velocity = -particle[i].Velocity 

                # Apply position limits
                particle[i].Position = np.maximum(particle[i].Position, VarMin)
                particle[i].Position = np.minimum(particle[i].Position, VarMax)

                # evaluation
                particle[i].Cost = fitness(particle[i].Position[0], Eload, G, T, Vw, inputs)

                # update personal best
                if particle[i].Cost < particle[i].BestCost:
                    particle[i].BestPosition = particle[i].Position
                    particle[i].BestCost = particle[i].Cost

                    # update global best
                    if particle[i].BestCost < GlobalBest["Cost"]:
                        GlobalBest["Position"] = particle[i].BestPosition
                        GlobalBest["Cost"] = particle[i].BestCost
        
            BestCost[it] = GlobalBest["Cost"]
            temp = 0
            for j in range(nPop):
                temp = temp + particle[j].BestCost
            MeanCost[it] = temp / nPop

            print("Run time = ", tt)
            print("Iteration = ", it)
            print("Best Cost = ", BestCost[it])
            print("Mean Cost = ", MeanCost[it])

            w = w*wdamp
    
        if GlobalBest["Cost"] < FinalBest["Cost"]:
            FinalBest["Cost"] = GlobalBest["Cost"]
            FinalBest["Position"] = GlobalBest["Position"]
            FinalBest["CostCurve"] = BestCost

    return FinalBest["Position"]



if __name__ == "__main__":
    main()
