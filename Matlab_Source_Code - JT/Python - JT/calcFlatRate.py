import numpy as np

def calcFlatRate(price):
    Cbuy = np.zeros(8760)
    for h in range(8760):
        Cbuy[h] = price

    return Cbuy

np.savetxt("Cbuy.csv", calcFlatRate(0.12))
