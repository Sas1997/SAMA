from getLoad import getLoad
import numpy as np

def calcMonthlyTieredRate(base_charge_tier_1, base_charge_tier_2, base_charge_tier_3, prices, tierMax, load, daysInMonth):
    Cbuy = np.zeros(8760)
    base_charge = np.zeros(12)

    hCount = 0
    for m in range(12):
        monthlyLoad = 0
        for h in range(24 * daysInMonth[m]):
            monthlyLoad += load[hCount]

            if monthlyLoad < tierMax[m][0]:
                Cbuy[hCount] = prices[m][0]
            elif monthlyLoad < tierMax[m][1]:
                Cbuy[hCount] = prices[m][1]
            else:
                Cbuy[hCount] = prices[m][2]
            hCount += 1

    return [Cbuy, base_charge]

########################################################################################################################

load = getLoad("Load.csv")

tiered = calcMonthlyTieredRate(
    1, 2, 3,
    [
        [1, 2, 3],
        [4, 5, 6],
        [7, 8, 9],
        [10, 11, 12],
        [13, 14, 15],
        [16, 17, 18],
        [1, 2, 3],
        [4, 5, 6],
        [7, 8, 9],
        [10, 11, 12],
        [13, 14, 15],
        [16, 17, 18]
    ],
    [
        [500, 1000, 2000],
        [500, 1000, 2000],
        [500, 1000, 2000],
        [500, 1000, 2000],
        [500, 1000, 2000],
        [600, 1200, 2000],
        [600, 1200, 2000],
        [600, 1200, 2000],
        [600, 1200, 2000],
        [600, 1200, 2000],
        [600, 1200, 2000],
        [500, 1000, 2000]
    ],
    load,
    [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
)
np.savetxt("Cbuy.csv", tiered[0])
np.savetxt("mc.csv", tiered[1])