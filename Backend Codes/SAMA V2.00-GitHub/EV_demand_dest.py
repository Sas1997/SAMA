# Distribute excess demand if it exceeds Pev_max
from numba import jit
@jit(nopython=True, fastmath=True)
def distribute_excess_demand(ev_demand, pev_max):
    """
    Distribute excess EV demand exceeding pev_max to previous hours.

    Parameters:
    ev_demand (numpy array): Array of EV demand values for each hour.
    pev_max (float): Maximum EV charge/discharge rate.

    Returns:
    numpy array: Adjusted EV demand array with excess demand distributed.
    """
    ev_demand = ev_demand.copy()  # Ensure we do not modify the original array
    for i in range(len(ev_demand)):
        if ev_demand[i] > pev_max:  # Check if demand exceeds Pev_max
            excess = ev_demand[i] - pev_max
            ev_demand[i] = pev_max

            # Distribute excess demand to previous hours
            j = i - 1
            while excess > 0 and j >= 0:
                available_capacity = max(pev_max - ev_demand[j], 0)
                charge = min(excess, available_capacity)
                ev_demand[j] += charge
                excess -= charge
                j -= 1

    return ev_demand
