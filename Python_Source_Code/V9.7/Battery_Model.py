import math


"""
Battery Model function
"""


def battery_model(Qmax, Nbatt, Q, InData):
    dt = 1  # the length of the time step[h]
    Q1 = InData.c * Q  # the available energy[kWh] in the storage at the beginning of the time step
    # Q the total amount of energy[kWh] in the storage at the beginning of the time step
    # Qmax is the total capacity of the storage bank[kWh]
    # Nbatt the number of batteries in the storage bank
    npexp = math.exp(-InData.k * dt)
    Pch_max1 = -InData.k * (-InData.c * Qmax + Q1 * npexp + Q * InData.c * (1 - npexp)) / (1 - npexp + InData.c *
                                                                                           (InData.k * dt - 1 + npexp))
    Pch_max2 = (1 - math.exp(-InData.alfa_battery * dt)) * (Qmax - Q) / dt
    Pch_max3 = Nbatt * InData.Imax * InData.Vnom / 1000

    Pdch_max = InData.k *(Q1 * npexp + Q * InData.c * (1 - npexp)) * math.sqrt(InData.ef_bat) / (1 - npexp + InData.c *
                                                                                            (InData.k * dt - 1 + npexp))

    Pch_max = min([Pch_max1, Pch_max2, Pch_max3]) / math.sqrt(InData.ef_bat)

    return Pch_max, Pdch_max
