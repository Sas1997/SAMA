from getLoad import getLoad
import numpy as np

def calcSeasonalTieredRate(base_charge_tier_1, base_charge_tier_2, base_charge_tier_3, prices, tierMax, load, months, daysInMonth):
    Cbuy = np.zeros(8760)
    base_charge = np.zeros(12)
    totalMonthlyLoad = np.zeros(12)
    hCount = 0

    for m in range(12):
        monthlyLoad = 0
        for h in range(24 * daysInMonth[m]):
            monthlyLoad += load(hCount)
            hCount += 1
        totalMonthlyLoad[m] = monthlyLoad

    hCount = 0
    for m in range(12):
        for h in range(24 * daysInMonth[m]):
            if totalMonthlyLoad[m] < tierMax[months[m]][0]:
                Cbuy[hCount] = prices[months[m]][0]
            elif totalMonthlyLoad[m] < tierMax[months[m]][1]:
                Cbuy[hCount] = prices[months[m]][1]
            else:
                Cbuy[hCount] = prices[months[m]][2]
            hCount += 1

        if totalMonthlyLoad[m] < tierMax[months[m]][0]:
            base_charge[m] = base_charge_tier_1
        elif totalMonthlyLoad[m] < tierMax[months[m]][1]:
            base_charge[m] = base_charge_tier_2
        else:
            base_charge[m] = base_charge_tier_3

    return [Cbuy, base_charge]


def calcSeasonalTieredRate1(base_charge_tier_1, base_charge_tier_2, base_charge_tier_3, prices, tierMax, load, months, daysInMonth):
    Cbuy = np.zeros(8760)
    TEMP_load = np.zeros(8760) # test load
    base_charge = np.zeros(12)

    hCount = 0
    for m in range(12):
        monthlyLoad = 0
        for h in range(24 * daysInMonth[m]):
            monthlyLoad += load[hCount]

            TEMP_load[hCount] = monthlyLoad

            if monthlyLoad < tierMax[months[m]][0]:
                Cbuy[hCount] = prices[months[m]][0]
            elif monthlyLoad < tierMax[months[m]][1]:
                Cbuy[hCount] = prices[months[m]][1]
            else:
                Cbuy[hCount] = prices[months[m]][2]

            hCount += 1

        if monthlyLoad < tierMax[months[m]][0]:
            base_charge[m] = base_charge_tier_1
        elif monthlyLoad < tierMax[months[m]][1]:
            base_charge[m] = base_charge_tier_2
        else:
            base_charge[m] = base_charge_tier_3

    return [Cbuy, base_charge, TEMP_load]

load = getLoad("Load.csv")

tiered = calcSeasonalTieredRate1(
    1, 2, 3,
    [[0.1, 0.2, 0.3],[1, 2, 3]],
    [[500, 1000, 2000],[600, 1200, 2000]],
    load,
    [0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 0],
    [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
)
np.savetxt("Cbuy.csv", tiered[0])
np.savetxt("mc.csv", tiered[1])
np.savetxt("addLoad.csv", tiered[2])
