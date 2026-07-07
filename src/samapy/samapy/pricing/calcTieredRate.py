import numpy as np


def calcTieredRate(tieredPrices, tierMax, Eload, daysInMonth):
    Cbuy = np.zeros(8760)
    hCount = 0

    for m in range(12):
        monthlyLoad = 0
        for h in range(24 * daysInMonth[m]):
            monthlyLoad += Eload[hCount]

            if monthlyLoad < tierMax[0]:
                Cbuy[hCount] = tieredPrices[0]
            elif monthlyLoad < tierMax[1]:
                Cbuy[hCount] = tieredPrices[1]
            else:
                Cbuy[hCount] = tieredPrices[2]

            hCount += 1

    return Cbuy