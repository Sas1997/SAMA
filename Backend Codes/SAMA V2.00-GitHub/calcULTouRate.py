from datetime import datetime, timedelta
import numpy as np

def calcULTouRate(year, onPrice, midPrice, offPrice, ultraLowPrice, onHours, midHours, ultraLowHours, season, daysInMonth,
                holidays, treat_special_days_as_offpeak):
    """
    Calculate Time of Use rates including ultra-low pricing periods.

    Parameters:
    -----------
    year : int
        Year for calculation
    onPrice : np.array
        On-peak prices [summer, winter]
    midPrice : np.array
        Mid-peak prices [summer, winter]
    offPrice : np.array
        Off-peak prices [summer, winter]
    ultraLowPrice : np.array
        Ultra-low prices [summer, winter]
    onHours : np.array
        On-peak hours [summer, winter]
    midHours : np.array
        Mid-peak hours [summer, winter]
    ultraLowHours : np.array
        Ultra-low hours [summer, winter]
    season : np.array
        Season definition for each month (0=winter, 1=summer)
    daysInMonth : list
        Days in each month
    holidays : list
        Holiday days (day of year)
    treat_special_days_as_offpeak : bool
        Whether to treat weekends/holidays as off-peak

    Returns:
    --------
    np.array
        8760 hourly pricing array
    """

    startDate = datetime(year, 1, 1)
    Cbuy = np.zeros(8760)
    tp = None
    tm = None
    tu = None  # ultra-low hours

    for m in range(12):
        t_start = 24 * sum(daysInMonth[0:m])
        t_end = 24 * sum(daysInMonth[0:m + 1])
        t_index = np.arange(t_start, t_end)

        if season[m] == 1:  # for summer
            if len(onHours[season[m] - 1]) > 0:
                tp = onHours[0]
            else:
                tp = None
            if len(midHours[season[m] - 1]) > 0:
                tm = midHours[0]
            else:
                tm = None
            if len(ultraLowHours[season[m] - 1]) > 0:
                tu = ultraLowHours[0]
            else:
                tu = None

            P_peak = onPrice[0]
            P_mid = midPrice[0]
            P_offpeak = offPrice[0]
            P_ultralow = ultraLowPrice[0]

        else:  # for winter
            if len(onHours[season[m]]) > 0:  # Fixed indexing
                tp = onHours[1]
            else:
                tp = None
            if len(midHours[season[m]]) > 0:  # Fixed indexing
                tm = midHours[1]
            else:
                tm = None
            if len(ultraLowHours[season[m]]) > 0:  # Fixed indexing
                tu = ultraLowHours[1]
            else:
                tu = None

            P_peak = onPrice[1]
            P_mid = midPrice[1]
            P_offpeak = offPrice[1]
            P_ultralow = ultraLowPrice[1]

        # Set all hours to off-peak by default
        Cbuy[t_index] = P_offpeak

        for d in range(daysInMonth[m]):
            # Apply ultra-low pricing first (lowest priority, can be overridden)
            if tu is not None:
                for hour in tu:
                    idx_ultra = t_index[hour] + 24 * d
                    if idx_ultra < len(Cbuy):
                        Cbuy[idx_ultra] = P_ultralow

            # Apply mid-peak pricing (medium priority)
            if tm is not None:
                for hour in tm:
                    idx_mid = t_index[hour] + 24 * d
                    if idx_mid < len(Cbuy):
                        Cbuy[idx_mid] = P_mid

            # Apply on-peak pricing (highest priority)
            if tp is not None:
                for hour in tp:
                    idx_peak = t_index[hour] + 24 * d
                    if idx_peak < len(Cbuy):
                        Cbuy[idx_peak] = P_peak

    # Handle special days (weekends and holidays)
    if treat_special_days_as_offpeak:
        for d in range(365):
            currentDate = startDate + timedelta(days=d)
            currentMonth = currentDate.month - 1  # months are 0 indexed in the season array

            if season[currentMonth] == 1:
                P_offpeak_special = offPrice[0]  # summer off-peak
                P_ultralow_special = ultraLowPrice[0]  # summer ultra-low
            else:
                P_offpeak_special = offPrice[1]  # winter off-peak
                P_ultralow_special = ultraLowPrice[1]  # winter ultra-low

            currentDayOfWeek = currentDate.weekday()  # Monday is 0 and Sunday is 6

            # Weekend treatment: 7 AM - 11 PM = off-peak, 11 PM - 7 AM = ultra-low
            if currentDayOfWeek == 5 or currentDayOfWeek == 6:  # Saturday or Sunday
                st = 24 * d
                ed = 24 * (d + 1)
                if ed <= len(Cbuy):
                    # Apply off-peak to 7 AM - 11 PM (hours 7-22)
                    for hour in range(7, 23):
                        hour_idx = st + hour
                        if hour_idx < len(Cbuy):
                            Cbuy[hour_idx] = P_offpeak_special

                    # Apply ultra-low to 11 PM - 7 AM (hours 23, 0-6)
                    for hour in [23, 0, 1, 2, 3, 4, 5, 6]:
                        hour_idx = st + hour
                        if hour_idx < len(Cbuy):
                            Cbuy[hour_idx] = P_ultralow_special

            # Holiday treatment: same as weekends
            if d + 1 in holidays:
                st = 24 * d
                ed = 24 * (d + 1)
                if ed <= len(Cbuy):
                    # Apply off-peak to 7 AM - 11 PM (hours 7-22)
                    for hour in range(7, 23):
                        hour_idx = st + hour
                        if hour_idx < len(Cbuy):
                            Cbuy[hour_idx] = P_offpeak_special

                    # Apply ultra-low to 11 PM - 7 AM (hours 23, 0-6)
                    for hour in [23, 0, 1, 2, 3, 4, 5, 6]:
                        hour_idx = st + hour
                        if hour_idx < len(Cbuy):
                            Cbuy[hour_idx] = P_ultralow_special

    return Cbuy