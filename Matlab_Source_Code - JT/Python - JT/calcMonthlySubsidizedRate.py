from getLoad import getLoad
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

standardPrices = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
subsidizedPrices = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1, 2, 3]
subsidyThreshold = [750, 750, 750, 1000, 1000, 1000, 1000, 1000, 1000, 750, 750, 750]
load = getLoad("Load.csv")
daysInMonth = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]

cost = calcMonthlyRate(standardPrices, subsidizedPrices, subsidyThreshold, load, daysInMonth)
np.savetxt("Cbuy.csv", cost[0])
