from getLoad import getLoad
import numpy as np

def calcSubsidizedRate(standardPrice, subsidizedPrice, subsidyThreshold, load, daysInMonth):
    Cbuy = np.zeros(8760)

    hCount = 0
    for m in range(12):
        monthlyLoad = 0
        hoursStart = hCount
        hoursEnd = hoursStart + (24 * daysInMonth[m])
        hoursRange = range(hoursStart, hoursEnd)

        for h in range(24 * daysInMonth[m]):
            monthlyLoad += load[hCount]
            hCount += 1

        if monthlyLoad <= subsidyThreshold:
            Cbuy[hoursRange] = subsidizedPrice
        else:
            Cbuy[hoursRange] = standardPrice

    return Cbuy

load = getLoad("Load.csv")
np.savetxt("Cbuy.csv", calcSubsidizedRate(2, 1, 1000, load, [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]))
