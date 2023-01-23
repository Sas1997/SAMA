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
