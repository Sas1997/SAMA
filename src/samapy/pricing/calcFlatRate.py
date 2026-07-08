import numpy as np
def calcFlatRate(flatPrice):
    Cbuy = np.array([flatPrice] * 8760)
    return Cbuy