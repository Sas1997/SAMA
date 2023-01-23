import numpy as np

def calcTouRate(onPrice, midPrice, offPrice, onHours, midHours, offHours, Month, Day, holidays):
    Cbuy = np.zeros(8760)
    for m in range(1, 13):
        t_start = (24 * np.sum(Day[0:m - 1]) + 1).astype(int)
        t_end = (24 * np.sum(Day[0:m])).astype(int)
        t_index = list(range(t_start, t_end + 1))
        t_index = np.array(t_index).astype(int)
        nt = len(t_index)

        if Month[m - 1] == 1:  # for summer
            tp = np.array(onHours[1]).astype(int)
            tm = np.array(midHours[1]).astype(int)
            toff = np.array(offHours[1]).astype(int)
            P_peak = onPrice[1]
            P_mid = midPrice[1]
            P_offpeak = offPrice[1]
        else:  # for winter
            tp = np.array(onHours[0]).astype(int)
            tm = np.array(midHours[0]).astype(int)
            toff = np.array(offHours[0]).astype(int)
            P_peak = onPrice[0]
            P_mid = midPrice[0]
            P_offpeak = offPrice[0]

        print(tp, tm, toff)

        Cbuy[t_index - 1] = P_offpeak  # set all hours to offpeak by default
        for d in range(1, Day[m - 1] + 1):
            idx0 = np.array(t_index[tp] + 24 * (d - 1))
            Cbuy[idx0 - 1] = P_peak
            idx1 = np.array(t_index[tm] + 24 * (d - 1))
            Cbuy[idx1 - 1] = P_mid

    for d in range(1, 365):
        if ((d - 1) % 7) >= 5:
            st = 24 * (d - 1) + 1
            ed = 24 * d
            Cbuy[range(st, ed)] = P_offpeak

    for d in range(1, 365):
        if d in holidays:
            st = 24 * (d - 1) + 1
            ed = 24 * d
            Cbuy[range(st, ed)] = P_offpeak

    return Cbuy
  
