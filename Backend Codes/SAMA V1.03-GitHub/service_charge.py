import numpy as np

def service_charge(daysInMonth, Eload_Previous, Limit_SC_1, SC_1, Limit_SC_2, SC_2, Limit_SC_3, SC_3, SC_4):
    totalmonthlyload = np.zeros((12, 1))
    hourCount = 0
    for m in range(12):
        monthlyLoad = 0
        for h in range(24 * daysInMonth[m]):
            monthlyLoad += Eload_Previous[hourCount]
            hourCount += 1
        totalmonthlyload[m, 0] = monthlyLoad

    max_monthly_load = max(totalmonthlyload)
    if max_monthly_load < Limit_SC_1:
        Service_charge = np.ones(12) * SC_1
    elif max_monthly_load < Limit_SC_2:
        Service_charge = np.ones(12) * SC_2
    elif max_monthly_load < Limit_SC_3:
        Service_charge = np.ones(12) * SC_3
    else:
        Service_charge = np.ones(12) * SC_4

    return Service_charge