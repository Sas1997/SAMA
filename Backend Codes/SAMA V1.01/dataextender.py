def dataextender(daysInMonth, Monthly_haverage_load):

    Eload = []
    for m in range(12):
        for h in range(24 * daysInMonth[m]):
            Eload.append(Monthly_haverage_load[m])

    return Eload