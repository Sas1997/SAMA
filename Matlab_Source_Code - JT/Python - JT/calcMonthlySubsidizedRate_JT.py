import numpy as np

def calcMonthlyRate(standardPrices, subsidizedPrices, subsidyThreshold, load, daysInMonth):
    Cbuy = np.zeros(8760)
    totalMonthlyLoad = np.zeros(12)

    hCount = 0
    for m in range(12):
        monthlyLoad = 0
        for h in range(24 * daysInMonth[m]):
            monthlyLoad += load[hCount]
            hCount += 1
        totalMonthlyLoad[m] = monthlyLoad

    hCount = 0
    for m in range(12):
        if totalMonthlyLoad[m] <= subsidyThreshold[m]:
            prices = subsidizedPrices[m]
        else:
            prices = standardPrices[m]

        for h in range(24 * daysInMonth[m]):
            Cbuy[hCount] = prices
            hCount += 1

    return Cbuy
