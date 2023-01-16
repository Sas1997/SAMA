import numpy as np

def calcMonthlyRate(prices, daysInMonth):
    Cbuy = np.zeros(8760)
    hCount = 0

    for m in range(12):
        for h in range(24 * daysInMonth[m]):
            Cbuy[hCount] = prices[m]
            hCount += 1

    return Cbuy

np.savetxt("Cbuy.csv", calcMonthlyRate([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12], [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]))