import numpy as np


def calcSeasonalTieredRate(seasonalTieredPrices, seasonalTierMax, Eload, season):
    daysInMonth = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
    Cbuy = np.zeros(8760)

    hCount = 0
    for m in range(12):
        monthlyLoad = 0
        for h in range(24 * daysInMonth[m]):
            monthlyLoad += Eload[hCount]

            if monthlyLoad < seasonalTierMax[season[m], 0]:
                Cbuy[hCount] = seasonalTieredPrices[season[m], 0]
            elif monthlyLoad < seasonalTierMax[season[m], 1]:
                Cbuy[hCount] = seasonalTieredPrices[season[m], 1]
            else:
                Cbuy[hCount] = seasonalTieredPrices[season[m], 2]
            hCount += 1

    return Cbuy