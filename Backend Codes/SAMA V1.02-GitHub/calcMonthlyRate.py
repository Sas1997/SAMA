import numpy as np
def calcMonthlyRate(monthlyPrices, daysInMonth):
    Cbuy = np.zeros(8760)
    hCount = 0
    for m in range(12):
        for h in range(24 * daysInMonth[m]):
            Cbuy[hCount] = monthlyPrices[m]
            hCount += 1
    return Cbuy