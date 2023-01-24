from getLoad import getLoad
import numpy as np

def calcSeasonalSubsidizedRate(standardPrices, subsidizedPrices, subsidyThreshold, load, months, daysInMonth):
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
        if totalMonthlyLoad[m] <= subsidyThreshold[months[m]]:
            prices = subsidizedPrices[months[m]]
        else:
            prices = standardPrices[months[m]]

        for h in range(24 * daysInMonth[m]):
            Cbuy[hCount] = prices
            hCount += 1

    return Cbuy

standardPrices = [1, 2]
subsidizedPrices = [0.1, 0.2]
subsidyThreshold = [700, 1000]
load = getLoad("Load.csv")
months = [0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0]
daysInMonth = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]

cost = calcSeasonalSubsidizedRate(standardPrices, subsidizedPrices, subsidyThreshold, load, months, daysInMonth)
np.savetxt("Cbuy.csv", cost)
