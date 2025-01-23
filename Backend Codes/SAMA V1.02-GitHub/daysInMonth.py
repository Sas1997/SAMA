import numpy as np

def daysInMonth(year):
    if year % 4 == 0:
        daysInMonth = np.array([31, 29, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31])
    else:
        daysInMonth = np.array([31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31])

    return daysInMonth