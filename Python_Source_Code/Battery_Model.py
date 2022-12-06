import math
import numpy as np
"""
Battery Model function
"""
def battery_model(
    Nbat, 
    Eb, 
    alfa_battery,
    c, 
    k, 
    Imax, 
    Vnom, 
    ef_bat, 
) -> (float, float):
    dt = 1         # the length of the time step [h]
    Q1 = c * Eb    # the available energy [kWh] in the storage at the beginning of the time step
    Q = Eb         # the total amount of energy [kWh] in the storage at the beginning of the time step
    Qmax = Nbat    # the total capacity of the storage bank [kWh]

    Pch_max1 = -(-k*c*Qmax+k*Q1*math.exp(-k*dt) + Q*k*c*(1-math.exp(-k*dt))) / (1-math.exp(-k*dt) + c*(k*dt-1+math.exp(-k*dt)))
    Pch_max2 = (1-math.exp(-alfa_battery*dt)) * (Qmax-Q) / dt
    Pch_max3 = Nbat * Imax * Vnom / 1000

    Pdch_max = (k*Q1*math.exp(-k*dt) + Q*k*c*(1-math.exp(-k*dt))) / (1-math.exp(-k*dt) + c*(k*dt-1+math.exp(-k*dt))) * np.sqrt(ef_bat)
    
    Pch_max = min([Pch_max1, Pch_max2, Pch_max3]) / np.sqrt(ef_bat)

    return Pdch_max, Pch_max
