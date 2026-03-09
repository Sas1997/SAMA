import numpy as np

def compute_ev_travel_energy(Eev, EV_p, SOC_dep, SOC_arr, C_ev):
    Eev_travel = np.zeros_like(Eev[0: 8760])
    travel_indices = np.where(EV_p == 0)[0]

    if len(travel_indices) > 0:
        start_idx = travel_indices[0]

        for i in range(1, len(travel_indices)):
            if travel_indices[i] != travel_indices[i - 1] + 1:
                end_idx = travel_indices[i - 1]
                E_travel_total = (SOC_dep - SOC_arr) * C_ev
                duration = end_idx - start_idx + 1
                decrement = E_travel_total / duration

                for j in range(duration):
                    Eev_travel[start_idx + j] = Eev[start_idx] - (j + 1) * decrement

                start_idx = travel_indices[i]

        end_idx = travel_indices[-1]
        E_travel_total = (SOC_dep - SOC_arr) * C_ev
        duration = end_idx - start_idx + 1
        decrement = E_travel_total / duration

        for j in range(duration):
            Eev_travel[start_idx + j] = Eev[start_idx] - (j + 1) * decrement

    return Eev_travel

def calculate_energy_consumption(Eev, EV_p):
    Pev_travel = np.zeros_like(EV_p, dtype=float)  # Match EV_p size
    for t in range(1, len(EV_p)):
        if EV_p[t] == 0:  # EV is traveling
            Pev_travel[t] = Eev[t] - Eev[t + 1]  # Energy consumption

    return Pev_travel