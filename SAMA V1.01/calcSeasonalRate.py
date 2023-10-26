import numpy as np

def calcSeasonalRate(seasonalPrices, season, daysInMonth):
    Cbuy = np.zeros(8760)
    hCount = 0

    for m in range(12):
        hoursStart = hCount
        hoursEnd = hoursStart + (24 * daysInMonth[m])
        hoursRange = np.array(range(hoursStart, hoursEnd))
        Cbuy[hoursRange] = seasonalPrices[season[m]]
        hCount = hoursEnd

    return Cbuy