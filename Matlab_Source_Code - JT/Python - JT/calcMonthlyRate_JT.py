import numpy as np

def calcMonthlyRate(prices, daysInMonth):
    Cbuy = np.zeros(8760)
    hCount = 0

    for m in range(12):
        for h in range(24 * daysInMonth[m]):
            Cbuy[hCount] = prices[m]
            hCount += 1

    return Cbuy
