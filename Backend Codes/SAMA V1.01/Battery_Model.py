from math import exp
from numba import jit
import numpy as np

"""
Battery Model function
"""


@jit(nopython=True, fastmath=True)
def battery_model(Cn_B, Nbat, Eb, alfa_battery, c, k, Imax, Vnom, ef_bat):
    dt = 1  # the length of the time step [h]
    Q1 = c * Eb  # the available energy [kWh] in the storage at the beginning of the time step
    Q=Eb    # the total amount of energy [kWh] in the storage at the beginning of the time step
    Qmax = Cn_B  # the total capacity of the storage bank [kWh]

    Pch_max1 = -(-k * c * Qmax + k * Q1 * exp(-k * dt) + Q * k * c * (1 - exp(-k * dt))) / (1 - exp(-k * dt) + c * (k * dt - 1 + exp(-k * dt)))
    Pch_max2 = (1 - exp(-alfa_battery * dt)) * (Qmax - Q) / dt
    Pch_max3 = Nbat * Imax * Vnom / 1000

    Pdch_max = (k * Q1 * exp(-k * dt) + Q * k * c * (1 - exp(-k * dt))) * np.sqrt(ef_bat) / (1 - exp(-k * dt) + c * (k * dt - 1 + exp(-k * dt)))

    Pch_max = min([Pch_max1, Pch_max2, Pch_max3]) / np.sqrt(ef_bat)

    return Pdch_max, Pch_max