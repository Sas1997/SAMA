from datetime import datetime, timedelta
import numpy as np

def calcTouRate(year, onPrice, midPrice, offPrice, onHours, midHours, offHours, season, daysInMonth, holidays):

  startDate = datetime(year, 1, 1)
  Cbuy = np.zeros(8760)

  for m in range(12):
    t_start = 24 * sum(daysInMonth[0:m])
    t_end = 24 * sum(daysInMonth[0:m + 1])
    t_index = np.arange(t_start, t_end)

    if season[m] == 1:  # for summer
      tp = onHours[0, :]
      tm = midHours[0, :]
      toff = offHours[0, :]
      P_peak = onPrice[0]
      P_mid = midPrice[0]
      P_offpeak = offPrice[0]

    else:  # for winter
      tp = onHours[1, :]
      tm = midHours[1, :]
      toff = offHours[1, :]
      P_peak = onPrice[1]
      P_mid = midPrice[1]
      P_offpeak = offPrice[1]

    Cbuy[t_index] = P_offpeak  # set all hours to off-peak by default

    for d in range(daysInMonth[m]):
      idx0 = t_index[tp-1] + 24 * d
      Cbuy[idx0] = P_peak
      idx1 = t_index[tm-1] + 24 * d
      Cbuy[idx1] = P_mid

  for d in range(365):
    currentDate = startDate + timedelta(days=d)
    currentDayOfWeek = currentDate.weekday()  # Monday is 0 and Sunday is 6

    if currentDayOfWeek == 5 or currentDayOfWeek == 6:  # Saturday or Sunday
      st = 24 * d
      ed = 24 * (d + 1)
      Cbuy[st:ed] = P_offpeak

    if d + 1 in holidays:
      st = 24 * d
      ed = 24 * (d + 1)
      Cbuy[st:ed] = P_offpeak

  return Cbuy