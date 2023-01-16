import numpy as np

def calcTieredRate(base_charge_tier_1, base_charge_tier_2, base_charge_tier_3, prices, tierMax, load, daysInMonth):
    Cbuy = np.zeros(8760)
    base_charge = np.zeros(12)
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
        for h in range(24 * daysInMonth[m]):
            if totalMonthlyLoad[m] < tierMax[0]:
                Cbuy[hCount] = prices[0]
            elif totalMonthlyLoad[m] < tierMax[1]:
                Cbuy[hCount] = prices[1]
            else:
                Cbuy[hCount] = prices[2]
            hCount += 1

        if totalMonthlyLoad[m] < tierMax[0]:
            base_charge[m] = base_charge_tier_1
        elif totalMonthlyLoad[m] < tierMax[1]:
            base_charge[m] = base_charge_tier_2
        else:
            base_charge[m] = base_charge_tier_3

    return [Cbuy, base_charge]
