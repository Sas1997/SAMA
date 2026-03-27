import numpy as np
import pandas as pd
import itertools
import math
from numba import njit
from samapy import get_content_path


# =============================================================================
# JIT-COMPILED KERNELS  (module-level so Numba can cache across calls)
# =============================================================================

@njit(cache=True)
def poly2d_nb(X, Y, coeffs, deg_x, deg_y):
    """
    2-D polynomial evaluation — identical term order to the original poly2d()
    but compiled to native code by Numba.
    cache=True writes the compiled binary to disk so subsequent runs skip
    the ~1-2 s recompilation cost.
    """
    Z = 0.0
    idx = 0
    for t in range(deg_x + deg_y + 1):
        px_min = t - deg_y
        if px_min < 0:
            px_min = 0
        px_max = deg_x
        if px_max > t:
            px_max = t
        for px in range(px_max, px_min - 1, -1):
            py = t - px
            if idx < len(coeffs):
                Z += coeffs[idx] * (X ** px) * (Y ** py)
            idx += 1
    return Z


@njit(cache=True)
def _heating_core(stacked_hs, stacked_hp,
                  To, Tid, To_norm, Tid_norm,
                  deg_x_hs, deg_y_hs, deg_x_hp, deg_y_hp,
                  load, kW_to_BTU):
    """
    Evaluate heating supply and power curves then interpolate to find
    actual power consumption. All inputs are plain scalars or 2-D
    float64 arrays — fully Numba-compatible.
    """
    Z_hs = np.empty(5)
    Z_hp = np.empty(5)
    for i in range(5):
        Z_hs[i] = poly2d_nb(To,      Tid,      stacked_hs[i], deg_x_hs, deg_y_hs)
        Z_hp[i] = poly2d_nb(To_norm, Tid_norm, stacked_hp[i], deg_x_hp, deg_y_hp)

    i_high = 0
    while i_high < 5 and Z_hs[i_high] < load:
        i_high += 1
    i_low = i_high - 1

    if load <= Z_hs[0]:
        COP      = Z_hs[0] / (Z_hp[0] * kW_to_BTU)
        Zhp_load = load / (COP * kW_to_BTU)
    elif load >= Z_hs[4]:
        COP      = Z_hs[4] / (Z_hp[4] * kW_to_BTU)
        Zhp_load = load / (COP * kW_to_BTU)
    else:
        Zhp_load = (Z_hp[i_low]
                    + (Z_hp[i_high] - Z_hp[i_low])
                    * (load - Z_hs[i_low]) / (Z_hs[i_high] - Z_hs[i_low]))
        COP = load / (Zhp_load * kW_to_BTU)

    return Zhp_load, COP


@njit(cache=True)
def _cooling_core(b_cs_b0, b_cs_b1, b_cs_b2,
                  b_cp_b0, b_cp_b1, b_cp_b2,
                  Tid, To, Tid_norm, To_norm, Tiw,
                  deg_x_cs, deg_y_cs, deg_x_cp, deg_y_cp,
                  load, kW_to_BTU):
    """
    Evaluate cooling supply and power curves then interpolate.
    Coefficient triplets (b0/b1/b2) are 2-D float64 arrays, shape (5, n_coeffs).
    The Tiw-polynomial is evaluated here to stay inside the JIT boundary.
    """
    Tiw2 = Tiw * Tiw
    Z_cs = np.empty(5)
    Z_cp = np.empty(5)
    for i in range(5):
        array_cs_i = b_cs_b0[i] + b_cs_b1[i] * Tiw + b_cs_b2[i] * Tiw2
        array_cp_i = b_cp_b0[i] + b_cp_b1[i] * Tiw + b_cp_b2[i] * Tiw2
        Z_cs[i] = poly2d_nb(Tid,      To,      array_cs_i, deg_x_cs, deg_y_cs)
        Z_cp[i] = poly2d_nb(Tid_norm, To_norm, array_cp_i, deg_x_cp, deg_y_cp)

    i_high = 0
    while i_high < 5 and Z_cs[i_high] < load:
        i_high += 1
    i_low = i_high - 1

    if load <= Z_cs[0]:
        COP      = Z_cs[0] / (Z_cp[0] * kW_to_BTU)
        Zcp_load = load / (COP * kW_to_BTU)
    elif load >= Z_cs[4]:
        COP      = Z_cs[4] / (Z_cp[4] * kW_to_BTU)
        Zcp_load = load / (COP * kW_to_BTU)
    else:
        Zcp_load = (Z_cp[i_low]
                    + (Z_cp[i_high] - Z_cp[i_low])
                    * (load - Z_cs[i_low]) / (Z_cs[i_high] - Z_cs[i_low]))
        COP = load / (Zcp_load * kW_to_BTU)

    return Zcp_load, COP


# =============================================================================
# MAIN MODEL FUNCTION
# =============================================================================

def Heat_Pump_Model(T, P, Hload, Cload):
    """
    Unified Bosch Heat Pump Model Function

    Parameters:
    -----------
    T : array        Hourly ambient temperature [C]
    P : array        Hourly ambient pressure [kPa]
    Hload : array    Hourly heating load [kW]
    Cload : array    Hourly cooling load [kW]

    Returns:
    --------
    power_hp_total   : array   Total hourly electricity consumption [kW]
    power_hp_heating : array   Heating mode electricity consumption [kW]
    power_hp_cooling : array   Cooling mode electricity consumption [kW]
    COP_hp_heating   : array   Hourly COP in heating mode
    COP_hp_cooling   : array   Hourly COP in cooling mode
    hp_model         : str     Selected heat pump model description
    HP_size          : int     Total heat pump system capacity [BTU/hr]
    """

    max_heating_load_hourly = np.amax(Hload)
    max_cooling_load_hourly = np.amax(Cload)

    # Heating set-points
    T_in_design_heating = 23   # indoor design temperature in winter [C]

    # Cooling set-points
    T_in_design_cooling = 22   # indoor design temperature in summer [C]
    w_in_design         = 0.005  # indoor design humidity ratio [kg/kg da]

    # Calculate indoor wet bulb temperature (vectorised — unchanged)
    P_sat_in = (0.623692418 + 0.0424692499 * T_in_design_cooling +
                0.00134403923  * T_in_design_cooling ** 2 +
                0.0000309447379 * T_in_design_cooling ** 3 +
                3.74294905E-07  * T_in_design_cooling ** 4)
    P_in      = P
    RH_in     = w_in_design * P_in / (w_in_design * P_sat_in + 0.622 * P_sat_in)
    RH_in_100 = RH_in * 100
    T_iwb_cooling = (T_in_design_cooling * np.arctan(0.151977 * (RH_in_100 + 8.313659) ** 0.5) +
                     np.arctan(T_in_design_cooling + RH_in_100) -
                     np.arctan(RH_in_100 - 1.676331) +
                     (0.00391838 * RH_in_100 ** 1.5) * np.arctan(0.023101 * RH_in_100) - 4.686035)

    # Heat pump sizes available [BTU/hr]
    y         = [24000, 36000, 48000, 60000]
    kW_to_BTU = 3412.142

    # -------------------------------------------------------------------------
    # OPTIMISATION 1: O(1) HP selection replacing the original O(max_units^4)
    # nested loops.
    #
    # All four unit sizes share GCD = 12 000 BTU/hr, so any achievable total
    # is a multiple of 12 000. The minimum total >= load is therefore
    # ceil(load / 12 000) * 12 000. A one-pass greedy then decomposes that
    # target into actual units — HP_size is identical to the original.
    # -------------------------------------------------------------------------
    def selectHP(demand_kW):
        load   = demand_kW * kW_to_BTU
        GCD    = 12_000
        target = max(int(math.ceil(load / GCD) * GCD), 24_000)
        T_val  = target // GCD   # integer >= 2

        n3  = T_val // 5
        rem = T_val % 5
        counts = [0, 0, 0, 0]
        if rem == 0:
            counts[3] = n3
        elif rem == 1:          # can't patch with a single smaller unit; swap
            counts[3] = n3 - 1  # one 60k → two 36k (net change: +12k = +1 GCD unit ✓)
            counts[1] = 2
        elif rem == 2:
            counts[3] = n3
            counts[0] = 1       # one 24 000
        elif rem == 3:
            counts[3] = n3
            counts[1] = 1       # one 36 000
        elif rem == 4:
            counts[3] = n3
            counts[2] = 1       # one 48 000

        return np.array([counts, y])

    # Select heat pumps based on peak loads
    matrix_heating = selectHP(max_heating_load_hourly)
    matrix_cooling = selectHP(max_cooling_load_hourly)

    # Filter and select final configuration (logic unchanged)
    mask_heating = matrix_heating[0, :] > 0
    mask_cooling = matrix_cooling[0, :] > 0

    filtered_heating = matrix_heating[:, mask_heating]
    filtered_cooling = matrix_cooling[:, mask_cooling]

    combined_size_heating = (filtered_heating[0, :] * filtered_heating[1, :]).sum()
    combined_size_cooling  = (filtered_cooling[0, :] * filtered_cooling[1, :]).sum()

    Final_HP_size_matrix = (filtered_heating
                            if combined_size_heating > combined_size_cooling
                            else filtered_cooling)

    HP_size = int((Final_HP_size_matrix[0, :] * Final_HP_size_matrix[1, :]).sum())

    # Build flat list of installed unit capacities
    Final_installed_units = []
    for count, cap in zip(Final_HP_size_matrix[0, :], Final_HP_size_matrix[1, :]):
        Final_installed_units.extend([int(cap)] * int(count))

    # Create model description string
    hp_model = "Bosch: " + ", ".join([f"{int(n)}x{int(cap)}BTU"
                                      for n, cap in zip(Final_HP_size_matrix[0],
                                                        Final_HP_size_matrix[1]) if n > 0])

    # Normalisation parameters (unchanged)
    X_range_h = np.array([30, 22.2, 19.4, 16.7, 13.9, 11.1, 8.3, 5.6, 2.8, 0,
                          -2.8, -5.6, -8.3, -11.1, -13.9, -16.7, -20])
    Y_range_h = np.array([15.6, 21.1, 23.9, 26.7])
    X_range_c = np.array([21.11, 23.89, 26.67, 29.44])
    Y_range_c = np.array([18.33, 23.89, 29.44, 35, 40.56, 46.11, 51.67])

    X_mean_h, X_std_h = np.mean(X_range_h), np.std(X_range_h, ddof=0)
    Y_mean_h, Y_std_h = np.mean(Y_range_h), np.std(Y_range_h, ddof=0)
    X_mean_c, X_std_c = np.mean(X_range_c), np.std(X_range_c, ddof=0)
    Y_mean_c, Y_std_c = np.mean(Y_range_c), np.std(Y_range_c, ddof=0)

    # -------------------------------------------------------------------------
    # OPTIMISATION 2: Read every Excel file exactly once up front via
    # get_content_path (identical calls to the original), store as contiguous
    # float64 arrays so they can be passed directly into @njit kernels.
    # -------------------------------------------------------------------------
    heating_cache = {}
    for size in [24000, 36000, 48000, 60000]:
        path   = get_content_path(f'HP_Bosch/H-{size}.xlsx')
        hdf_hs = pd.read_excel(path, sheet_name='Heating Supply')
        hdf_p  = pd.read_excel(path, sheet_name='Power')
        heating_cache[size] = dict(
            stacked_hs = np.ascontiguousarray(
                np.vstack([hdf_hs.iloc[:, i].to_numpy(dtype=float) for i in range(1, 6)])),
            stacked_hp = np.ascontiguousarray(
                np.vstack([hdf_p.iloc[:,  i].to_numpy(dtype=float) for i in range(1, 6)])),
            deg_x_hs   = int(hdf_hs.iloc[0, 6]),
            deg_y_hs   = int(hdf_hs.iloc[0, 7]),
            deg_x_hp   = int(hdf_p.iloc[0, 6]),
            deg_y_hp   = int(hdf_p.iloc[0, 7]),
        )

    cooling_cache = {}
    for size in [24000, 36000, 48000, 60000]:
        path   = get_content_path(f'HP_Bosch/C-{size}.xlsx')
        cdf_cs = pd.read_excel(path, sheet_name='Cooling Supply')
        cdf_p  = pd.read_excel(path, sheet_name='Power')
        # Store b-coefficient triplets as stacked (5 × n_coeffs) float64 arrays.
        # The Tiw-polynomial (b0 + b1*Tiw + b2*Tiw²) is evaluated per-hour
        # inside the @njit kernel, so only the raw coefficients are cached.
        cooling_cache[size] = dict(
            b_cs_b0 = np.ascontiguousarray(
                np.vstack([cdf_cs.iloc[:, 1+3*i].to_numpy(dtype=float) for i in range(5)])),
            b_cs_b1 = np.ascontiguousarray(
                np.vstack([cdf_cs.iloc[:, 2+3*i].to_numpy(dtype=float) for i in range(5)])),
            b_cs_b2 = np.ascontiguousarray(
                np.vstack([cdf_cs.iloc[:, 3+3*i].to_numpy(dtype=float) for i in range(5)])),
            b_cp_b0 = np.ascontiguousarray(
                np.vstack([cdf_p.iloc[:,  1+3*i].to_numpy(dtype=float) for i in range(5)])),
            b_cp_b1 = np.ascontiguousarray(
                np.vstack([cdf_p.iloc[:,  2+3*i].to_numpy(dtype=float) for i in range(5)])),
            b_cp_b2 = np.ascontiguousarray(
                np.vstack([cdf_p.iloc[:,  3+3*i].to_numpy(dtype=float) for i in range(5)])),
            deg_x_cs = int(cdf_cs.iloc[0, 16]),
            deg_y_cs = int(cdf_cs.iloc[0, 17]),
            deg_x_cp = int(cdf_p.iloc[0, 16]),
            deg_y_cp = int(cdf_p.iloc[0, 17]),
        )

    # Thin Python wrappers — unpack cache dict and forward to @njit kernels
    def heating_model(size, l, To, Tid):
        if l == 0:
            return 0, np.nan
        d        = heating_cache[size]
        To_norm  = (To  - X_mean_h) / X_std_h
        Tid_norm = (Tid - Y_mean_h) / Y_std_h
        return _heating_core(
            d['stacked_hs'], d['stacked_hp'],
            float(To), float(Tid), To_norm, Tid_norm,
            d['deg_x_hs'], d['deg_y_hs'], d['deg_x_hp'], d['deg_y_hp'],
            l * kW_to_BTU, kW_to_BTU,
        )

    def cooling_model(size, l, To, Tid, Tiw):
        if l == 0:
            return 0, np.nan
        d        = cooling_cache[size]
        Tid_norm = (Tid - X_mean_c) / X_std_c
        To_norm  = (To  - Y_mean_c) / Y_std_c
        return _cooling_core(
            d['b_cs_b0'], d['b_cs_b1'], d['b_cs_b2'],
            d['b_cp_b0'], d['b_cp_b1'], d['b_cp_b2'],
            float(Tid), float(To), Tid_norm, To_norm, float(Tiw),
            d['deg_x_cs'], d['deg_y_cs'], d['deg_x_cp'], d['deg_y_cp'],
            l * kW_to_BTU, kW_to_BTU,
        )

    # -------------------------------------------------------------------------
    # OPTIMISATION 3: Pre-build dispatch look-up table once.
    #
    # The original dispatch_heat_pumps() rebuilt and searched all combinations
    # on every one of the 8 760 hourly calls, even though Final_installed_units
    # never changes. We generate every combination once, sort by
    # (unit-count, total-capacity), and at runtime do a single linear scan —
    # the first entry whose total >= hourly load is the answer.
    # -------------------------------------------------------------------------
    _n = len(Final_installed_units)
    _dispatch_table = []
    for _r in range(1, _n + 1):
        for _combo in itertools.combinations(range(_n), _r):
            _total = sum(Final_installed_units[_i] for _i in _combo)
            _dispatch_table.append((_r, _total, _combo))
    _dispatch_table.sort(key=lambda x: (x[0], x[1]))

    def dispatch_heat_pumps(load):
        hourly_load_Btu = kW_to_BTU * load
        active = set()
        for _r, _total, _combo in _dispatch_table:
            if _total >= hourly_load_Btu:
                active = set(_combo)
                break
        return [[cap, (1 if i in active else 0)]
                for i, cap in enumerate(Final_installed_units)]

    # -------------------------------------------------------------------------
    # WARM-UP: trigger JIT compilation before the main loop so the first
    # simulated hour is not slowed by a one-time ~1-2 s compile.
    # -------------------------------------------------------------------------
    _dummy = np.ones((5, 10), dtype=np.float64)
    _heating_core(_dummy, _dummy, 0.0, 0.0, 0.0, 0.0, 2, 2, 2, 2, 1.0, kW_to_BTU)
    _cooling_core(_dummy, _dummy, _dummy, _dummy, _dummy, _dummy,
                  0.0, 0.0, 0.0, 0.0, 0.0, 2, 2, 2, 2, 1.0, kW_to_BTU)

    # -------------------------------------------------------------------------
    # Main hourly loop — logic identical to the original
    # -------------------------------------------------------------------------
    power_hp_heating = np.zeros(len(Hload))
    power_hp_cooling = np.zeros(len(Cload))
    COP_hp_heating   = np.full(len(Hload), np.nan)
    COP_hp_cooling   = np.full(len(Cload), np.nan)

    for i in range(len(Hload)):

        if T[i] < -20:
            power_hp_heating[i] = 0  # supplementary heating system is required
        else:
            matrix_h  = dispatch_heat_pumps(Hload[i])
            m         = [row[0] for row in matrix_h]  # nominal capacities of selected heat pumps
            n         = [row[1] for row in matrix_h]  # on/off flags
            denominator = sum(m[j] * n[j] for j in range(len(m)))  # total active nominal capacity
            c         = [(m[k] * n[k]) / denominator for k in range(len(m))]  # contribution share
            l         = [c[u] * Hload[i] for u in range(len(c))]              # demand share per unit
            p_h       = [heating_model(m[y], l[y], T[i], T_in_design_heating)[0]
                         for y in range(len(l))]
            p_h_clean = np.array([np.nan if x is None else x for x in p_h], dtype=float)
            power_hp_heating[i] = np.nansum(p_h_clean)
            if power_hp_heating[i] > 0 and Hload[i] > 0:
                COP_hp_heating[i] = Hload[i] / power_hp_heating[i]

        if T[i] < 18.33:
            power_hp_cooling[i] = 0  # ventilation can address the load
        else:
            matrix_c  = dispatch_heat_pumps(Cload[i])
            m         = [row[0] for row in matrix_c]  # nominal capacities of selected heat pumps
            n         = [row[1] for row in matrix_c]  # on/off flags
            denominator = sum(m[j] * n[j] for j in range(len(m)))  # total active nominal capacity
            c         = [(m[k] * n[k]) / denominator for k in range(len(m))]  # contribution share
            l         = [c[u] * Cload[i] for u in range(len(c))]              # demand share per unit
            p_c       = [cooling_model(m[y], l[y], T[i], T_in_design_cooling, T_iwb_cooling[i])[0]
                         for y in range(len(l))]
            p_c_clean = np.array([np.nan if x is None else x for x in p_c], dtype=float)
            power_hp_cooling[i] = np.nansum(p_c_clean)
            if power_hp_cooling[i] > 0 and Cload[i] > 0:
                COP_hp_cooling[i] = Cload[i] / power_hp_cooling[i]

    power_hp_heating = np.nan_to_num(power_hp_heating, nan=0)
    power_hp_cooling = np.nan_to_num(power_hp_cooling, nan=0)

    power_hp_total = power_hp_heating + power_hp_cooling

    return power_hp_total, power_hp_heating, power_hp_cooling, COP_hp_heating, COP_hp_cooling, hp_model, HP_size
