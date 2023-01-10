import numpy as np
class Solution():
    def __init__(self):
        self.BestCost = None
        self.BestSol = None
        self.CostCurve = None


class Particle():
    def __init__(self):
        self.Position = 0
        self.Cost = 0
        self.Velocity = 0

        # to represent personal best
        self.BestPosition = 0
        self.BestCost = 0