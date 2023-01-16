import numpy as np

def calcSeasonalRate(prices, months, daysInMonth):
    Cbuy = np.zeros(8760)
    hCount = 0

    for m in range(12):
        hoursStart = hCount
        hoursEnd = hoursStart + (24 * daysInMonth[m])
        hoursRange = list(range(hoursStart, hoursEnd))
        Cbuy[hoursRange] = prices[months[m]]
        hCount = hoursEnd

    return Cbuy

np.savetxt("Cbuy.csv", calcSeasonalRate([0.08, 0.12], [0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0], [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]))