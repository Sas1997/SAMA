from datetime import datetime, timedelta
import numpy as np
import pandas as pd

def determine_EV_presence(year, Tout, Tin, holidays, treat_special_days_as_home):
    """
    Determines the EV presence for a given year, considering weekends and holidays.

    Parameters:
        year (int): The target year.
        Tout (int): Hour when EV leaves home.
        Tin (int): Hour when EV returns home.
        holidays (list): List of holiday dates as day-of-year integers (1-365/366).

    Returns:
        np.ndarray: Array of size 8760 indicating EV presence (1 = at home, 0 = away).
    """
    hours_per_year = 8760 if (year % 4 != 0 or (year % 100 == 0 and year % 400 != 0)) else 8784
    EV_p = np.ones(hours_per_year, dtype=int)  # Default: EV is at home

    start_date = datetime(year, 1, 1)

    for d in range(365 + (hours_per_year == 8784)):  # Loop over days
        current_date = start_date + timedelta(days=d)
        current_day_of_week = current_date.weekday()  # Monday=0, Sunday=6
        day_of_year = d + 1  # Day of the year (1 to 365/366)

        # Check if it's a weekday (Monday-Friday) and not a holiday
        if treat_special_days_as_home and current_day_of_week < 5 and day_of_year not in holidays or not treat_special_days_as_home:
            tt = np.arange(24 * d, 24 * (d + 1))  # Hours for the day
            EV_p[tt[Tout:Tin]] = 0  # EV is away during work hours

    return EV_p

# # Example Usage
# year = 2025
# Tout = 0  # EV leaves home at 8 AM
# Tin = 7  # EV returns home at 6 PM
# holidays = [1, 51, 97, 100, 142, 182, 219, 248, 273, 282, 315, 359, 360]
# EV_presence = determine_EV_presence(year, Tout, Tin, holidays)
#
# data = {'EV P': EV_presence}
# df = pd.DataFrame(data)
# df.to_csv('output/data/EVP.csv', index=False)