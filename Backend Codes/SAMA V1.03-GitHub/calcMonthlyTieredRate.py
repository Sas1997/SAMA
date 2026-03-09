import numpy as np


def calcMonthlyTieredRate(monthlyTieredPrices, monthlyTierLimits, Eload):
    Cbuy = np.zeros(8760)
    daysInMonth = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]

    hCount = 0
    for m in range(12):
        monthlyLoad = 0
        for h in range(24 * daysInMonth[m]):
            monthlyLoad += Eload[hCount]
            if monthlyLoad < monthlyTierLimits[m][0]:
                Cbuy[hCount] = monthlyTieredPrices[m][0]
            elif monthlyLoad < monthlyTierLimits[m][1]:
                Cbuy[hCount] = monthlyTieredPrices[m][1]
            else:
                Cbuy[hCount] = monthlyTieredPrices[m][2]
            hCount += 1
    return Cbuy