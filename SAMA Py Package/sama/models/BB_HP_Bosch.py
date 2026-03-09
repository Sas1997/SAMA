import numpy as np
import pandas as pd
import itertools
from sama import get_content_path

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
        demand_BTU = demand_kW * kW_to_BTU

        # Check if single unit can cover the load
        larger_units = [cap for cap in y if cap >= demand_BTU]
        if larger_units:
            chosen = min(larger_units)
            idx = y.index(chosen)
            n = [0, 0, 0, 0]
            n[idx] = 1
            result = np.array([n, y])
            return result

        # Multiple units required
        best_combo = None
        min_total = float('inf')
        search_range = range(0, 6)

        for n1, n2, n3, n4 in itertools.product(search_range, repeat=4):
            total_capacity = n1 * y[0] + n2 * y[1] + n3 * y[2] + n4 * y[3]

            if total_capacity >= demand_BTU and total_capacity < min_total and total_capacity != 0:
                min_total = total_capacity
                best_combo = (n1, n2, n3, n4)

        if best_combo:
            result = np.array([best_combo, y])
            return result
        else:
            return None

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
            return 0, 1.0

        load = l * kW_to_BTU

        # Load coefficient files based on size
        file_map = {24000: get_content_path('HP_Bosch/H-24000.xlsx'), 36000: get_content_path('HP_Bosch/H-36000.xlsx'),
                    48000: get_content_path('HP_Bosch/H-48000.xlsx'), 60000: get_content_path('HP_Bosch/H-60000.xlsx')}

        if size not in file_map:
            return 0, 1.0

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
            return 0, 1.0

        load = l * kW_to_BTU

        file_map = {24000: get_content_path('HP_Bosch/C-24000.xlsx'), 36000: get_content_path('HP_Bosch/C-36000.xlsx'),
                    48000: get_content_path('HP_Bosch/C-48000.xlsx'), 60000: get_content_path('HP_Bosch/C-60000.xlsx')}

        if size not in file_map:
            return 0, 1.0

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

    # Calculate hourly power consumption and COP
    power_hp_heating = np.zeros(len(Hload))
    power_hp_cooling = np.zeros(len(Cload))
    COP_hp_heating = np.ones(len(Hload))
    COP_hp_cooling = np.ones(len(Cload))

    for m, n in zip(Final_HP_size_matrix[0], Final_HP_size_matrix[1]):
        temp_power_h = np.zeros(len(Hload))
        temp_power_c = np.zeros(len(Cload))
        temp_cop_h = np.ones(len(Hload))
        temp_cop_c = np.ones(len(Cload))

        for i in range(len(Hload)):
            if T[i] < -20:
                temp_power_h[i] = 0
                temp_cop_h[i] = 1.0
            else:
                temp_power_h[i], temp_cop_h[i] = heating_model(int(n), Hload[i], T[i], T_in_design_heating)

            if T[i] < 18.33:
                temp_power_c[i] = 0
                temp_cop_c[i] = 1.0
            else:
                temp_power_c[i], temp_cop_c[i] = cooling_model(int(n), Cload[i], T[i], T_in_design_cooling,
                                                               T_iwb_cooling[i])

        temp_power_h = np.nan_to_num(temp_power_h, nan=0)
        temp_power_c = np.nan_to_num(temp_power_c, nan=0)
        temp_cop_h = np.nan_to_num(temp_cop_h, nan=1.0)
        temp_cop_c = np.nan_to_num(temp_cop_c, nan=1.0)

        power_hp_heating += m * temp_power_h
        power_hp_cooling += m * temp_power_c

        # Weighted average for COP (only where there's load)
        mask_h = Hload > 0
        mask_c = Cload > 0
        COP_hp_heating[mask_h] = temp_cop_h[mask_h]
        COP_hp_cooling[mask_c] = temp_cop_c[mask_c]

    power_hp_total = power_hp_heating + power_hp_cooling

    return power_hp_total, power_hp_heating, power_hp_cooling, COP_hp_heating, COP_hp_cooling, hp_model, HP_size