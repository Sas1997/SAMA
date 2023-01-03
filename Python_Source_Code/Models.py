import numpy as np

from Fitness import fitness

class Solution():
    __slots__ = ("BestCost", "BestSol", "CostCurve")
    def __init__(self):
        self.BestCost = None
        self.BestSol = None
        self.CostCurve = None


class Particle():
    __slots__ = ("Position", "Cost", "Velocity", "BestPosition", "BestCost")
    def __init__(self, VarSize, VarMax, VarMin, nVar, globals):
        iterable = (np.random.uniform(VarMin[var], VarMax[var]) for var in range(nVar))
        self.Position = np.fromiter(iterable, float, nVar)
        self.Cost = fitness(self.Position, globals["Eload"], globals["G"], globals["T"], globals["Vw"], globals["ins_parameter"])  
        self.Velocity = np.zeros(VarSize)

        # to represent personal best
        self.BestPosition = self.Position
        self.BestCost = self.Cost