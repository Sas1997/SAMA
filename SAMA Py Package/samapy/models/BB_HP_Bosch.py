import numpy as np
import pandas as pd
import itertools
import math
from samapy import get_content_path

def Heat_Pump_Model(T, P, Hload, Cload):
    """
    Unified Bosch Heat Pump Model Function

    Parameters:
    -----------
    T : array
        Hourly ambient temperature [C]
    P : array
        Hourly ambient pressure [kPa]
    Hload : array
        Hourly heating load [kW]
    Cload : array
        Hourly cooling load [kW]

    Returns:
    --------
    power_hp_total : array
        Total hourly electricity consumption [kW]
    power_hp_heating : array
        Heating mode electricity consumption [kW]
    power_hp_cooling : array
        Cooling mode electricity consumption [kW]
    COP_hp_heating : array
        Hourly COP in heating mode
    COP_hp_cooling : array
        Hourly COP in cooling mode
    hp_model : str
        Selected heat pump model description
    HP_size : int
        Total heat pump system capacity [BTU/hr]
    """

    max_heating_load_hourly = np.amax(Hload)
    max_cooling_load_hourly = np.amax(Cload)

    # Heating set-points
    T_in_design_heating = 23  # indoor design temperature in winter [C]

    # Cooling set-points
    T_in_design_cooling = 22  # indoor design temperature in summer [C]
    w_in_design = 0.005  # indoor design humidity ratio [kg/kg da]

    # Calculate indoor wet bulb temperature
    P_sat_in = (0.623692418 + 0.0424692499 * T_in_design_cooling +
                0.00134403923 * (T_in_design_cooling) ** 2 +
                0.0000309447379 * (T_in_design_cooling) ** 3 +
                3.74294905E-07 * (T_in_design_cooling) ** 4)
    P_in = P
    RH_in = w_in_design * P_in / (w_in_design * P_sat_in + 0.622 * P_sat_in)
    RH_in_100 = RH_in * 100
    T_iwb_cooling = (T_in_design_cooling * np.arctan(0.151977 * (RH_in_100 + 8.313659) ** 0.5) +
                     np.arctan(T_in_design_cooling + RH_in_100) -
                     np.arctan(RH_in_100 - 1.676331) +
                     (0.00391838 * (RH_in_100) ** 1.5) * np.arctan(0.023101 * RH_in_100) - 4.686035)

    # Heat pump sizes available [BTU/hr]
    y = [24000, 36000, 48000, 60000]
    kW_to_BTU = 3412.142

    # Heat pump selection function
    def selectHP(demand_kW):

        # Finds the minimum combination of heat pumps with the smallest total installed capacity that covers the load

        max_cap = max(y)
        min_cap = min(y)
        load = demand_kW * kW_to_BTU
        max_units = math.ceil(load / min_cap) + 20  # upper bound on number of units needed

        best_solution = None
        best_total = float("inf")

        for n1 in range(max_units):
            for n2 in range(max_units):
                for n3 in range(max_units):
                    for n4 in range(max_units):
                        total = (
                            n1 * y[0] +
                            n2 * y[1] +
                            n3 * y[2] +
                            n4 * y[3]
                        )
                        if total >= load and total < best_total:
                            best_total = total
                            best_solution = (n1, n2, n3, n4)

        result = np.array([best_solution, y])
        return result

    # Select heat pumps based on loads
    matrix_heating = selectHP(max_heating_load_hourly)
    matrix_cooling = selectHP(max_cooling_load_hourly)

    # Filter and select final configuration
    mask_heating = matrix_heating[0, :] > 0
    mask_cooling = matrix_cooling[0, :] > 0

    filtered_heating = matrix_heating[:, mask_heating]
    filtered_cooling = matrix_cooling[:, mask_cooling]

    combined_size_heating = (filtered_heating[0, :] * filtered_heating[1, :]).sum()
    combined_size_cooling = (filtered_cooling[0, :] * filtered_cooling[1, :]).sum()

    if combined_size_heating > combined_size_cooling:
        Final_HP_size_matrix = filtered_heating
    else:
        Final_HP_size_matrix = filtered_cooling

    HP_size = int((Final_HP_size_matrix[0, :] * Final_HP_size_matrix[1, :]).sum())

    # Build flat list of installed unit capacities
    Final_installed_units = []
    for count, cap in zip(Final_HP_size_matrix[0, :], Final_HP_size_matrix[1, :]):
        Final_installed_units.extend([int(cap)] * int(count))

    # Create model description string
    hp_model = "Bosch: " + ", ".join([f"{int(n)}x{int(cap)}BTU"
                                      for n, cap in zip(Final_HP_size_matrix[0],
                                                        Final_HP_size_matrix[1]) if n > 0])

    # Normalization parameters
    X_range_h = np.array([30, 22.2, 19.4, 16.7, 13.9, 11.1, 8.3, 5.6, 2.8, 0,
                          -2.8, -5.6, -8.3, -11.1, -13.9, -16.7, -20])
    Y_range_h = np.array([15.6, 21.1, 23.9, 26.7])
    X_range_c = np.array([21.11, 23.89, 26.67, 29.44])
    Y_range_c = np.array([18.33, 23.89, 29.44, 35, 40.56, 46.11, 51.67])

    X_mean_h, X_std_h = np.mean(X_range_h), np.std(X_range_h, ddof=0)
    Y_mean_h, Y_std_h = np.mean(Y_range_h), np.std(Y_range_h, ddof=0)
    X_mean_c, X_std_c = np.mean(X_range_c), np.std(X_range_c, ddof=0)
    Y_mean_c, Y_std_c = np.mean(Y_range_c), np.std(Y_range_c, ddof=0)

    def normalize(x_entry, x_mean, x_std):
        return (x_entry - x_mean) / x_std

    def poly2d(X, Y, coeffs, deg_x, deg_y):
        terms = []
        for t in range(deg_x + deg_y + 1):
            px_min = max(0, t - deg_y)
            px_max = min(deg_x, t)
            for px in range(px_max, px_min - 1, -1):
                py = t - px
                terms.append((px, py))

        length = len(terms)
        if len(coeffs) < length:
            coeffs = np.pad(coeffs, (0, length - len(coeffs)))

        Z = 0.0
        for c, (px, py) in zip(coeffs, terms):
            Z += c * (X ** px) * (Y ** py)
        return Z

    # Heating model function
    def heating_model(size, l, To, Tid):
        if l == 0:
            return 0, np.nan

        load = l * kW_to_BTU

        # Load coefficient files based on size
        file_map = {24000: get_content_path('HP_Bosch/H-24000.xlsx'), 36000: get_content_path('HP_Bosch/H-36000.xlsx'),
                    48000: get_content_path('HP_Bosch/H-48000.xlsx'), 60000: get_content_path('HP_Bosch/H-60000.xlsx')}

        if size not in file_map:
            return 0, np.nan

        hdf_hs = pd.read_excel(file_map[size], sheet_name='Heating Supply')
        hdf_p = pd.read_excel(file_map[size], sheet_name='Power')

        X_degree_hs = int(hdf_hs.iloc[0, 6])
        Y_degree_hs = int(hdf_hs.iloc[0, 7])
        X_degree_hp = int(hdf_p.iloc[0, 6])
        Y_degree_hp = int(hdf_p.iloc[0, 7])

        arrays_hs = [hdf_hs.iloc[:, i].to_numpy() for i in range(1, 6)]
        arrays_hp = [hdf_p.iloc[:, i].to_numpy() for i in range(1, 6)]

        Z_hs = np.zeros(5)
        Z_hp = np.zeros(5)

        for i in range(5):
            Z_hs[i] = poly2d(To, Tid, arrays_hs[i], X_degree_hs, Y_degree_hs)
            Z_hp[i] = poly2d(normalize(To, X_mean_h, X_std_h),
                             normalize(Tid, Y_mean_h, Y_std_h),
                             arrays_hp[i], X_degree_hp, Y_degree_hp)

        i_high = np.searchsorted(Z_hs, load)
        i_low = i_high - 1

        if load <= Z_hs[0]:
            COP = Z_hs[0] / (Z_hp[0] * kW_to_BTU)
            Zhp_load = load / (COP * kW_to_BTU)
        elif load >= Z_hs[-1]:
            COP = Z_hs[-1] / (Z_hp[-1] * kW_to_BTU)
            Zhp_load = load / (COP * kW_to_BTU)
        else:
            Zhp_load = Z_hp[i_low] + (Z_hp[i_high] - Z_hp[i_low]) * (load - Z_hs[i_low]) / (Z_hs[i_high] - Z_hs[i_low])
            COP = load / (Zhp_load * kW_to_BTU)

        return Zhp_load, COP

    # Cooling model function
    def cooling_model(size, l, To, Tid, Tiw):
        if l == 0:
            return 0, np.nan

        load = l * kW_to_BTU

        file_map = {24000: get_content_path('HP_Bosch/C-24000.xlsx'), 36000: get_content_path('HP_Bosch/C-36000.xlsx'),
                    48000: get_content_path('HP_Bosch/C-48000.xlsx'), 60000: get_content_path('HP_Bosch/C-60000.xlsx')}

        if size not in file_map:
            return 0, np.nan

        cdf_cs = pd.read_excel(file_map[size], sheet_name='Cooling Supply')
        cdf_p = pd.read_excel(file_map[size], sheet_name='Power')

        X_degree_cs = int(cdf_cs.iloc[0, 16])
        Y_degree_cs = int(cdf_cs.iloc[0, 17])
        X_degree_cp = int(cdf_p.iloc[0, 16])
        Y_degree_cp = int(cdf_p.iloc[0, 17])

        coeffs_cs = cdf_cs.iloc[:, 0]
        coeffs_cp = cdf_p.iloc[:, 0]

        result_cs = pd.DataFrame({'Labels': coeffs_cs})
        result_cp = pd.DataFrame({'Labels': coeffs_cp})

        for i in range(5):
            b0_cs = cdf_cs.iloc[:, 1 + 3 * i]
            b1_cs = cdf_cs.iloc[:, 2 + 3 * i]
            b2_cs = cdf_cs.iloc[:, 3 + 3 * i]
            b0_cp = cdf_p.iloc[:, 1 + 3 * i]
            b1_cp = cdf_p.iloc[:, 2 + 3 * i]
            b2_cp = cdf_p.iloc[:, 3 + 3 * i]

            result_cs[f'Y{i + 1}'] = b0_cs + b1_cs * Tiw + b2_cs * (Tiw ** 2)
            result_cp[f'Y{i + 1}'] = b0_cp + b1_cp * Tiw + b2_cp * (Tiw ** 2)

        arrays_cs = [result_cs.iloc[:, i].to_numpy() for i in range(1, 6)]
        arrays_cp = [result_cp.iloc[:, i].to_numpy() for i in range(1, 6)]

        Z_cs = np.zeros(5)
        Z_cp = np.zeros(5)

        for i in range(5):
            Z_cs[i] = poly2d(Tid, To, arrays_cs[i], X_degree_cs, Y_degree_cs)
            Z_cp[i] = poly2d(normalize(Tid, X_mean_c, X_std_c),
                             normalize(To, Y_mean_c, Y_std_c),
                             arrays_cp[i], X_degree_cp, Y_degree_cp)

        i_high = np.searchsorted(Z_cs, load)
        i_low = i_high - 1

        if load <= Z_cs[0]:
            COP = Z_cs[0] / (Z_cp[0] * kW_to_BTU)
            Zcp_load = load / (COP * kW_to_BTU)
        elif load >= Z_cs[-1]:
            COP = Z_cs[-1] / (Z_cp[-1] * kW_to_BTU)
            Zcp_load = load / (COP * kW_to_BTU)
        else:
            Zcp_load = Z_cp[i_low] + (Z_cp[i_high] - Z_cp[i_low]) * (load - Z_cs[i_low]) / (Z_cs[i_high] - Z_cs[i_low])
            COP = load / (Zcp_load * kW_to_BTU)

        return Zcp_load, COP

    # Dispatch function - decides which heat pumps operate each hour based on actual load
    def dispatch_heat_pumps(installed_units, load):

        # Decides which heat pump must operate on each hour based on actual load on that hour

        kW_to_BTU = 3412.142  # rate we need to multiply kW demand by to get Btu/h
        hourly_load_Btu = kW_to_BTU * load
        n = len(installed_units)
        best_solution = None
        best_unit_count = float("inf")
        best_total_capacity = float("inf")

        for r in range(1, n + 1):
            for combo_indices in itertools.combinations(range(n), r):

                total = sum(installed_units[i] for i in combo_indices)

                if total >= hourly_load_Btu:

                    if r < best_unit_count:
                        best_solution = combo_indices
                        best_unit_count = r
                        best_total_capacity = total

                    elif r == best_unit_count and total < best_total_capacity:
                        best_solution = combo_indices
                        best_total_capacity = total

            if best_solution is not None:
                break

        dispatch_matrix = []  # building dispatch matrix

        for i, cap in enumerate(installed_units):

            if i in best_solution:
                dispatch_matrix.append([cap, 1])
            else:
                dispatch_matrix.append([cap, 0])

        return dispatch_matrix

    # Calculate hourly power consumption and COP
    power_hp_heating = np.zeros(len(Hload))
    power_hp_cooling = np.zeros(len(Cload))
    COP_hp_heating = np.full(len(Hload), np.nan)
    COP_hp_cooling = np.full(len(Cload), np.nan)

    for i in range(len(Hload)):

        if T[i] < -20:
            power_hp_heating[i] = 0  # supplementary heating system is required
        else:
            matrix_h = dispatch_heat_pumps(Final_installed_units, Hload[i])  # list and count of actual operating heat pumps based on the actual hourly load
            m = [row[0] for row in matrix_h]  # nominal capacities of selected heat pumps
            n = [row[1] for row in matrix_h]  # corresponding count of selected heat pumps
            denominator = sum(m[j] * n[j] for j in range(len(m)))  # total active nominal capacity
            c = [(m[k] * n[k]) / denominator for k in range(len(m))]  # contribution share of each heat pump
            l = [c[u] * Hload[i] for u in range(len(c))]  # actual demand share of each heat pump
            p_h = [heating_model(m[y], l[y], T[i], T_in_design_heating)[0] for y in range(len(l))]  # actual power consumption of each heat pump
            p_h_clean = np.array([np.nan if x is None else x for x in p_h], dtype=float)
            power_hp_heating[i] = np.nansum(p_h_clean)
            if power_hp_heating[i] > 0 and Hload[i] > 0:
                COP_hp_heating[i] = Hload[i] / power_hp_heating[i]

        if T[i] < 18.33:
            power_hp_cooling[i] = 0  # ventilation can address the load
        else:
            matrix_c = dispatch_heat_pumps(Final_installed_units, Cload[i])  # list and count of actual operating heat pumps based on the actual hourly load
            m = [row[0] for row in matrix_c]  # nominal capacities of selected heat pumps
            n = [row[1] for row in matrix_c]  # corresponding count of selected heat pumps
            denominator = sum(m[j] * n[j] for j in range(len(m)))  # total active nominal capacity
            c = [(m[k] * n[k]) / denominator for k in range(len(m))]  # contribution share of each heat pump
            l = [c[u] * Cload[i] for u in range(len(c))]  # actual demand share of each heat pump
            p_c = [cooling_model(m[y], l[y], T[i], T_in_design_cooling, T_iwb_cooling[i])[0] for y in range(len(l))]  # actual power consumption of each heat pump
            p_c_clean = np.array([np.nan if x is None else x for x in p_c], dtype=float)
            power_hp_cooling[i] = np.nansum(p_c_clean)
            if power_hp_cooling[i] > 0 and Cload[i] > 0:
                COP_hp_cooling[i] = Cload[i] / power_hp_cooling[i]

    power_hp_heating = np.nan_to_num(power_hp_heating, nan=0)
    power_hp_cooling = np.nan_to_num(power_hp_cooling, nan=0)

    power_hp_total = power_hp_heating + power_hp_cooling

    return power_hp_total, power_hp_heating, power_hp_cooling, COP_hp_heating, COP_hp_cooling, hp_model, HP_size
