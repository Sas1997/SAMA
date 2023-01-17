import numpy as np

def calcSeasonalTieredRate(base_charge_tier_1, base_charge_tier_2, base_charge_tier_3, prices, tierMax, load, months, daysInMonth):
    Cbuy = np.zeros(8760)
    base_charge = np.zeros(12)

    hCount = 0
    for m in range(12):
        monthlyLoad = 0
        for h in range(24 * daysInMonth[m]):
            monthlyLoad += load[hCount]
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

    return [Cbuy, base_charge]
