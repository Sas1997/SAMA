from numba import jit
import numpy as np
"""
Battery Model function
"""

@jit(nopython=True, fastmath=True)
def KiBaM(dt, Pch, Pdch, Cn_B, Nbat, Eb, SOC_min, SOC_max, alfa_battery_leadacid, c, k, Ich_max_leadacid, Vnom_leadacid, ef_bat):
    P = Pch + Pdch
    Q1 = Eb * c * np.exp(-k * dt) + (((Eb * k * c - P) * (1 - np.exp(-k * dt))) / k) - ((P * c * (k * dt - 1 + np.exp(-k * dt))) / k) # the available energy [kWh] in the storage at the beginning of the time step
    Q2 = Eb * (1 - c) * np.exp(-k * dt) + ((Eb * (1 - c)) * (1 - np.exp(-k * dt))) - ((P * (1 - c) * (k * dt - 1 + np.exp(-k * dt))) / k)
    Q = Q1 + Q2   # the total amount of energy [kWh] in the storage at the beginning of the time step
    Qmax = Cn_B  # the total capacity of the storage bank [kWh]

    Pch_max1 = -(-k * c * Qmax + k * Q1 * np.exp(-k * dt) + Q * k * c * (1 - np.exp(-k * dt))) / (1 - np.exp(-k * dt) + c * (k * dt - 1 + np.exp(-k * dt)))
    Pch_max2 = (1 - np.exp(-alfa_battery_leadacid * dt)) * (Qmax - Q) / dt
    Pch_max3 = Nbat * Ich_max_leadacid * Vnom_leadacid / 1000
    Pch_max4 = ((SOC_max * Cn_B) - Eb) / dt

    Pch_max = min([Pch_max1, Pch_max2, Pch_max3, Pch_max4]) / np.sqrt(ef_bat)

    SOC = Eb / Cn_B if (Cn_B != 0 and not np.isnan(Cn_B)) else 0

    if np.round(SOC, 4) == SOC_min:
        Pdch_max = 0
    else:
        Pdch_max = (k * Q1 * np.exp(-k * dt) + Q * k * c * (1 - np.exp(-k * dt))) * np.sqrt(ef_bat) / (1 - np.exp(-k * dt) + c * (k * dt - 1 + np.exp(-k * dt)))

    return Pdch_max, Pch_max


@jit(nopython=True, fastmath=True)
def IdealizedBattery(dt, SOC_min, SOC_max, alfa_battery_Li_ion, Nbat, Eb, Cn_B, Ich_max_Li_ion, Idch_max_Li_ion, Vnom_Li_ion, ef_bat):
    Q = Eb
    Qmax = Cn_B  # the total capacity of the storage bank [kWh]

    Pch_max1 = (1 - (np.exp(-alfa_battery_Li_ion * dt))) * (Qmax - Q) / dt
    Pch_max2 = Nbat * Ich_max_Li_ion * Vnom_Li_ion / 1000
    Pch_max3 = ((SOC_max * Cn_B) - Eb) / dt

    Pch_max = min([Pch_max1, Pch_max2, Pch_max3]) / np.sqrt(ef_bat)

    SOC = Eb / Cn_B if (Cn_B != 0 and not np.isnan(Cn_B)) else 0
    if np.round(SOC, 4) == SOC_min:
        Pdch_max = 0
    else:
        Pdch_max = np.sqrt(ef_bat) * Vnom_Li_ion * Idch_max_Li_ion * Nbat / 1000

    return Pdch_max, Pch_max