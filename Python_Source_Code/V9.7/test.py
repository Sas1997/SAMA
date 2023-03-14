import numpy as np
from InputData import InData
import matplotlib.pyplot as plt

from Fitness import fitness
from Results import results
from Utility_results import Utility_results
from Fitness import fitness
from PSO import PSO
from time import time


x = np.array([2, 10, 22, 10, 22.5])
t1=time()
z=fitness(x)
print(time()-t1)
# Psell = results(x)
# Utility_results(Psell)
print(z)
plt.show()
