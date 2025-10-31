import numpy as np
from Battery_Model import KiBaM, IdealizedBattery
from numba import jit
from EV_demand_dest import distribute_excess_demand


@jit(nopython=True, fastmath=True, cache=True)
def EMS_EV(Lead_acid, Li_ion, Ich_max_Li_ion, Idch_max_Li_ion, Vnom_Li_ion, ef_bat_Li, Q_lifetime_Li, alfa_battery_Li_ion, Ppv, Pwt, Eload, Cn_B, Nbat, Pn_DG, NT, SOC_max, SOC_min, SOC_initial, n_I, Grid, Cbuy, Csell, a, b, R_DG, TL_DG, MO_DG, Pinv_max, LR_DG, C_fuel, Pbuy_max, Psell_max, R_B, Q_lifetime_leadacid, self_discharge_rate, self_discharge_rate_ev, alfa_battery_leadacid, c, k_lead_acid, Ich_max_leadacid, Vnom_leadacid, ef_bat_leadacid, Tin, Tout, C_ev, Pev_max, SOCe_initial, SOC_dep, SOC_arr, SOCe_min, SOCe_max, n_e, Q_lifetime_ev, EV_p, R_EVB):
    # tdep = Tout

    if Li_ion == 1:
        ef_bat = ef_bat_Li
        Q_lifetime = Q_lifetime_Li

    if Lead_acid == 1:
        ef_bat = ef_bat_leadacid
        Q_lifetime = Q_lifetime_leadacid

    Eb = np.zeros(NT + 1)
    Pch = np.zeros(NT)
    Pdch = np.zeros(NT)
    Ech = np.zeros(NT)
    Edch = np.zeros(NT)
    Pdg = np.zeros(NT)
    Edump = np.zeros(NT)
    Ens = np.zeros(NT)
    Psell = np.zeros(NT)
    Pbuy = np.zeros(NT)
    Pinv = np.zeros(NT)
    Pdch_max = np.zeros(NT)
    Pch_max = np.zeros(NT)
    price_dg = np.zeros(NT)

    Ebmax = SOC_max * Cn_B
    Ebmin = SOC_min * Cn_B
    Eb[0] = SOC_initial * Cn_B
    dt = 1

    Eev = np.zeros(NT + 1)
    Eev_ch = np.zeros(NT)
    Eev_dch = np.zeros(NT)
    Pev_ch = np.zeros(NT)
    Pev_dch = np.zeros(NT)
    Ech_req = np.zeros(NT)
    Ee_min = SOCe_min * C_ev
    Ee_max = SOCe_max * C_ev
    Ee_dep = SOC_dep * C_ev
    Ee_0 = SOC_arr * C_ev

    Eev[0] = SOCe_initial * C_ev

    if Grid == 0:
        Pbuy_max = 0
        Psell_max = 0

    P_RE = Ppv + Pwt
    if sum(P_RE + Grid) == 0:
        Pdg_min = LR_DG * Pn_DG
    else:
        Pdg_min = 0.01 * Pn_DG

    # Battery Wear Cost
    Cbw = R_B * Cn_B / (Nbat * Q_lifetime * np.sqrt(ef_bat)) if Cn_B > 0 else np.inf
    Cevw = R_EVB / (Q_lifetime_ev * np.sqrt(n_e)) if C_ev > 0 else np.inf

    # DG fixed cost
    cc_gen = (b * Pn_DG * C_fuel) + ((R_DG * Pn_DG) / (TL_DG)) + MO_DG
    # DG marginal cost
    mar_gen = a * C_fuel
    # DG cost for cases
    for t in range(NT):
        price_dg[t] = ((cc_gen / Eload[t]) + mar_gen) if (Eload[t] != 0 and not np.isnan(Eload[t])) else np.inf

    t = np.arange(0, NT)  # Python range is 0-based and exclusive at the end
    ind = (EV_p[t] == 1) & (np.roll(EV_p, -1) == 0)
    # Calculate EV demand
    EV_dem = ind * (SOC_dep - SOC_arr) * C_ev
    EV_dem = distribute_excess_demand(EV_dem, Pev_max)


    # Simple price thresholds based on percentiles (approximated efficiently)
    buy_min = np.min(Cbuy)
    buy_max = np.max(Cbuy)
    sell_min = np.min(Csell)
    sell_max = np.max(Csell)

    # Approximate percentile thresholds (much faster than sorting)
    buy_range = buy_max - buy_min
    sell_range = sell_max - sell_min

    buy_threshold_25 = buy_min + 0.25 * buy_range  # Approximate 25th percentile
    sell_threshold_75 = sell_min + 0.75 * sell_range  # Approximate 75th percentile

    # Battery efficiency and cost parameters
    round_trip_efficiency = ef_bat * ef_bat

    for t in range(NT):

        if Li_ion == 1:
            Pdch_max[t], Pch_max[t] = IdealizedBattery(dt, SOC_min, SOC_max, alfa_battery_Li_ion, Nbat, Eb[t], Cn_B, Ich_max_Li_ion, Idch_max_Li_ion, Vnom_Li_ion, ef_bat)

        if Lead_acid == 1:
            Pdch_max[t], Pch_max[t] = KiBaM(dt, Pch[t], Pdch[t], Cn_B, Nbat, Eb[t], SOC_min, SOC_max, alfa_battery_leadacid, c, k_lead_acid, Ich_max_leadacid, Vnom_leadacid, ef_bat)

        Eev_ch[t] = EV_p[t] * max(0, Ee_max - Eev[t]) / n_e
        Eev_dch[t] = EV_p[t] * max(0, Eev[t] - Ee_min) * n_e

        if Grid == 0:
            Tch_min = int(np.ceil(max(0, Ee_dep - Eev[t]) / (n_e * Pev_max)))
            Ech_req[t] = 0

            if EV_p[t] == 1 and np.any(EV_p[t:t + Tch_min + 1] == 0):  # EV in home and need to charge
                Eev_dch[t] = 0
                Eev_ch[t] = 0
                Ech_req[t] = max(0, Ee_dep - Eev[t]) / (n_e * Tch_min)

            elif EV_p[t] == 1 and EV_p[t + 1] == 0:  # EV in home in last hour
                Eev_dch[t] = 0
                Eev_ch[t] = 0
                Ech_req[t] = max(0, Ee_dep - Eev[t]) / n_e

        if Grid == 1:
            if EV_p[t] == 1:  # EV is at home

                # STEP 1: Calculate energy needed to reach departure level
                energy_needed = max(0, Ee_dep - Eev[t])

                # STEP 2: Dynamically find actual departure time
                t_dep = -1
                for future_t in range(t + 1, NT):
                    if EV_p[future_t] == 0:
                        t_dep = future_t
                        break

                # STEP 3: Set lookahead window based on actual departure
                if t_dep == -1 or t_dep <= t:
                    optimal_lookahead = 1  # fallback if EV never leaves
                    t_dep = t + 1  # safe assumption for calculations below
                else:
                    optimal_lookahead = t_dep - t

                # STEP 4: Calculate remaining time and logic
                remaining_hours = max(1, t_dep - t)

                if energy_needed <= 0:
                    Ech_req[t] = 0  # No charging needed for departure
                    Eev_dch[t] = EV_p[t] * max(0, (Eev[t] - Ee_dep) * n_e)

                    # ENHANCED: Smart off-peak charging for V2G arbitrage
                    # Check if we have capacity and if conditions are right for profitable V2G
                    current_soc = Eev[t] / C_ev
                    available_capacity = max(0, SOCe_max - current_soc)

                    if available_capacity > 0.05:  # At least 5% capacity available
                        # Look ahead for profitable selling opportunities
                        future_sell_opportunities = 0.0
                        avg_future_sell_price = 0.0

                        # Check next optimal_lookahead hours for good selling prices
                        future_prices = []
                        for tau in range(t + 1, min(NT, t + optimal_lookahead)):
                            if Csell[tau] > Cevw * 1.1:  # Profitable selling threshold
                                future_prices.append(Csell[tau])

                        if len(future_prices) > 0:
                            avg_future_sell_price = np.mean(np.array(future_prices))

                            # Calculate potential profit from off-peak charging + future selling
                            round_trip_efficiency = n_e * n_e
                            break_even_buy_price = avg_future_sell_price * round_trip_efficiency

                            # If current buy price is significantly below break-even, charge for arbitrage
                            if Cbuy[t] < break_even_buy_price * 0.9:  # 10% profit margin minimum
                                arbitrage_profit_margin = (avg_future_sell_price * round_trip_efficiency - Cbuy[t]) / Cbuy[t]

                                if arbitrage_profit_margin > 0.15:  # 15% minimum profit margin
                                    # Calculate how much to charge for arbitrage
                                    max_arbitrage_charge = available_capacity * C_ev / n_e

                                    # Size the arbitrage charging based on profit opportunity
                                    arbitrage_multiplier = min(1.0, arbitrage_profit_margin / 0.3)  # Scale with profit

                                    optimal_arbitrage_charge = max_arbitrage_charge * arbitrage_multiplier
                                    optimal_arbitrage_charge = min(optimal_arbitrage_charge, Pev_max)

                                    # Only do arbitrage if we have sufficient time
                                    if remaining_hours >= 6:  # Need at least 6 hours for strategy
                                        Eev_ch[t] = max(Eev_ch[t], optimal_arbitrage_charge)
                else:
                    min_charging_hours = max(1, int(np.ceil(energy_needed / (n_e * Pev_max))))

                    if min_charging_hours >= remaining_hours:
                        Ech_req[t] = min(energy_needed / (n_e * remaining_hours), Pev_max)
                        Eev_dch[t] = 0
                    else:
                        lookahead_hours = min(remaining_hours, optimal_lookahead)
                        window = Cbuy[t:t + lookahead_hours]
                        flexibility_ratio = (remaining_hours - min_charging_hours) / max(1, remaining_hours)

                        # Look ahead for future RE surplus
                        future_RE_surplus = 0.0
                        for tau in range(t, min(NT, t + lookahead_hours)):
                            surplus = Ppv[tau] + Pwt[tau] - (Eload[tau] / n_I)
                            if surplus > 0:
                                cbuy_future = Cbuy[t + 1: t + lookahead_hours]
                                if len(cbuy_future) > 0 and Csell[tau] < np.min(cbuy_future):
                                    future_RE_surplus += surplus * n_I
                                else:
                                    break

                        RE_charging_energy = min(future_RE_surplus * dt, energy_needed)
                        grid_energy_needed = energy_needed - RE_charging_energy

                        sorted_hours = np.argsort(window)
                        best_hours = sorted_hours[:min_charging_hours]
                        in_best_hours = 0 in best_hours

                        if in_best_hours:
                            Ech_req[t] = min(grid_energy_needed / (n_e * min_charging_hours), Pev_max)
                            Eev_dch[t] = 0
                        elif flexibility_ratio < 0.3:
                            Ech_req[t] = min(grid_energy_needed / (n_e * remaining_hours), Pev_max)
                            Eev_dch[t] = 0
                        else:
                            Ech_req[t] = 0
                            safe_discharge = max(0, Eev[t] - Ee_dep - (energy_needed * 0.1))
                            Eev_dch[t] = EV_p[t] * min(max(0, Eev[t] - Ee_min), safe_discharge) * n_e

                        # If we have extra capacity and it's a good price, charge more for future arbitrage
                        current_soc_after_required = (Eev[t] + Ech_req[t] * n_e) / C_ev
                        remaining_capacity = max(0, SOCe_max - current_soc_after_required)

                        if remaining_capacity > 0.05 and remaining_hours >= min_charging_hours + 3:
                            # Look for future selling opportunities
                            future_sell_prices = []
                            for tau in range(t_dep, min(NT, t_dep + 24)):  # After departure
                                if tau < NT - 1 and EV_p[tau] == 1:  # When EV is back home
                                    if Csell[tau] > Cevw * 1.1:
                                        future_sell_prices.append(Csell[tau])

                            if len(future_sell_prices) > 0:
                                avg_future_sell = np.mean(np.array(future_sell_prices))
                                current_buy_adjusted = Cbuy[t] / (n_e * n_e)  # Account for round-trip losses

                                if avg_future_sell > current_buy_adjusted * 1.2:  # 20% profit potential
                                    extra_arbitrage_charge = remaining_capacity * C_ev / n_e * 0.5  # 50% of remaining capacity
                                    extra_arbitrage_charge = min(extra_arbitrage_charge, Pev_max - Ech_req[t])

                                    if extra_arbitrage_charge > 0:
                                        Ech_req[t] += extra_arbitrage_charge
            else:
                Ech_req[t] = 0
                Eev_dch[t] = 0

        if EV_p[t] == 1 and EV_p[t + 1] == 0:
            tdep = t + 1

        Ech_req[t] = min(Ech_req[t], Pev_max)
        Eev_ch[t] = min(Eev_ch[t], Pev_max)
        Eev_dch[t] = min(Eev_dch[t], Pev_max)

        PL_dc = (Eload[t] / n_I) + Ech_req[t]  # DC Load with EV charge demand

        if P_RE[t] >= PL_dc:  # if PV+Pwt greater than load  (battery should charge)
            Psur_dc = P_RE[t] - PL_dc

            # EV charge
            if Ech_req[t] == 0:
                Pev_ch[t] = min(Eev_ch[t], Psur_dc)
                Pev_ch[t] = min(Pev_ch[t], Pev_max)
                Psur_dc -= Pev_ch[t]
            else:
                Pev_ch[t] = Ech_req[t]

            # Battery charge power calculated based on surPlus energy and battery empty  capacity
            Eb_dch = (Ebmax - Eb[t]) / ef_bat
            Pch[t] = min(Eb_dch, Psur_dc)
            # Battery maximum charge power limit
            Pch[t] = min(Pch[t], Pch_max[t])  # Pch<=Pch_max

            Psur_AC = n_I * (Psur_dc - Pch[t])  # Surplus Energy

            Psell[t] = min(Psur_AC, Psell_max)  # Psell<=Psell_max
            Psell[t] = min(max(0, Pinv_max - PL_dc), Psell[t])

            Edump[t] = n_I * (P_RE[t] - Pch[t] - Pev_ch[t]) - Eload[t] - Psell[t]

        # if load greater than PV+Pwt
        else:
            Eb_dch = min(Pdch_max[t], max(0, Eb[t] - Ebmin) * ef_bat)
            Eb_ch = min(Pch_max[t], max(0, Ebmax - Eb[t]) / ef_bat)

            Price = np.array([Cbuy[t], price_dg[t], Cbw, Cevw])
            Tdc = np.array([n_I, n_I, 1, 1])

            Pmax = np.array([Pbuy_max, Pn_DG, Eb_dch, Eev_dch[t]])
            Pmin = np.array([-Psell_max, Pdg_min, -Eb_ch, -Eev_ch[t]])

            if Pdg_min >= (Eload[t] + Psell_max + (Ech_req[t] + Eev_ch[t] + Eb_ch) / n_I):
                Pmin = np.array([-Psell_max, 0 * Pdg_min, -Eb_ch, -Eev_ch[t]])

            if Ech_req[t] > 0:
                Price = np.delete(Price, 3)
                Tdc = np.delete(Tdc, 3)
                Pmax = np.delete(Pmax, 3)
                Pmin = np.delete(Pmin, 3)
                Pev_ch[t] = Ech_req[t]

            sorted_indices = np.argsort(Price)  # Source Priority set

            Edef_DC = P_RE[t] - Ech_req[t]  # Energy unbalance in DC Side
            Edef_AC0 = Eload[t] - min(Pinv_max, n_I * Edef_DC) * (Edef_DC > 0) + min(
                Pinv_max, -Edef_DC / n_I) * (Edef_DC < 0)

            ns = len(Pmax)
            P = np.zeros(ns)
            k = 0

            while abs(Edef_AC0) > 1e-3 and k < ns:
                i = sorted_indices[k]  # Source number
                if Tdc[i] == 1:  # Source is DC (Bat or EV)

                    # gen>Load                 Load>gen
                    P[i] = Edef_AC0 * n_I * (Edef_AC0 < 0) + (Edef_AC0 / n_I) * (Edef_AC0 > 0)
                    P[i] = max(Pmin[i], min(Pmax[i], P[i]))  # Source Limit

                    Pin = P_RE[t] - Ech_req[t] + np.sum(P[Tdc == 1])  # Inverter Power (to AC Side)
                    Pin_excess = np.sign(Pin) * max(0, abs(Pin) - Pinv_max)

                    P[i] -= Pin_excess
                    P[i] = max(Pmin[i], min(Pmax[i], P[i]))  # Source Limit

                    Edef_DC = P_RE[t] - Ech_req[t] + np.sum(P[Tdc == 1])  # Energy unbalance in DC Side
                    Edef_AC = Eload[t] - np.sum(P[Tdc != 1]) - min(Pinv_max, n_I * Edef_DC) * (Edef_DC > 0) + min(Pinv_max, -Edef_DC / n_I) * (Edef_DC < 0)

                else:  # Source is AC (Grid or DG)
                    P[i] = Edef_AC0
                    P[i] = max(Pmin[i], min(Pmax[i], P[i]))  # Source Limit

                    Edef_DC = P_RE[t] - Ech_req[t] + np.sum(P[Tdc == 1])  # Energy unbalance in DC Side
                    Edef_AC = Eload[t] - np.sum(P[Tdc != 1]) - min(Pinv_max, n_I * Edef_DC) * (Edef_DC > 0) + min(Pinv_max, -Edef_DC / n_I) * (Edef_DC < 0)

                if (i == 1) and (Edef_AC < 0):
                    temp = sorted_indices[0]
                    sorted_indices[0] = 1
                    sorted_indices[k] = temp
                    P[np.concatenate((np.array([0]), np.arange(2, len(P))))] = 0
                    k = 0
                    # Calculate Edef_DC (Energy unbalance in DC side)
                    Edef_DC = P_RE[t] - Ech_req[t] + np.sum(P[Tdc == 1])
                    # Calculate Edef_AC (Energy unbalance in AC side)
                    Edef_AC = (Eload[t] - np.sum(P[Tdc != 1]) - min(Pinv_max, n_I * Edef_DC) * (Edef_DC > 0) + min(Pinv_max, -Edef_DC / n_I) * (Edef_DC < 0))

                k += 1
                Edef_AC0 = Edef_AC

            Pbuy[t] = P[0] * (P[0] > 0)
            Psell[t] = - P[0] * (P[0] < 0)
            Pdg[t] = P[1]

            # FIXED: Simplified battery power assignment
            battery_power = P[2]
            if battery_power > 0:  # Discharging
                Pdch[t] = battery_power
                Pch[t] = 0
            else:  # Charging
                Pdch[t] = 0
                Pch[t] = -battery_power

            if Ech_req[t] == 0:
                Pev_dch[t] = P[3] * (P[3] > 0)
                Pev_ch[t] = - P[3] * (P[3] < 0)

            # ========================================
            # POST-OPTIMIZATION BES ARBITRAGE CHECK
            # ========================================

            # Check for additional arbitrage charging opportunity after main optimization
            if Grid == 1 and Cn_B > 0:
                remaining_battery_capacity = max(0, (Ebmax - Eb[t]) / ef_bat - Pch[t])
                remaining_grid_buy_capacity = max(0, Pbuy_max - Pbuy[t])

                # Calculate BES arbitrage opportunity
                current_battery_soc = Eb[t] / Cn_B
                available_charge_capacity = max(0, (SOC_max - current_battery_soc) * Cn_B / ef_bat)

                if (remaining_battery_capacity > 0.01 and remaining_grid_buy_capacity > 0.01 and available_charge_capacity > 0.01):

                    # Simple opportunity assessment
                    is_low_buy_price = Cbuy[t] < buy_threshold_25

                    # Look ahead for selling opportunities (simplified - check next 24 hours only)
                    max_future_sell_price = sell_threshold_75  # Default threshold
                    lookahead_end = min(NT, t + 24)

                    for tau in range(t + 1, lookahead_end):
                        if Csell[tau] > max_future_sell_price:
                            max_future_sell_price = Csell[tau]

                    # Simple profitability check
                    potential_profit = max_future_sell_price * round_trip_efficiency - Cbuy[t] - Cbw
                    is_profitable = potential_profit > Cbw * 0.15  # 15% minimum profit margin

                    # Simple arbitrage decision
                    if is_low_buy_price and is_profitable:
                        # Calculate charging intensity based on opportunity quality
                        price_advantage = (buy_threshold_25 - Cbuy[t]) / max(buy_range * 0.25, 1e-6)
                        price_advantage = max(0, min(1, price_advantage))

                        profit_ratio = potential_profit / max(Cbw, 1e-6)
                        profit_strength = min(1.0, profit_ratio / 0.15)  # Normalize to 15% baseline

                        # Combined opportunity score (simplified)
                        opportunity_score = (price_advantage + profit_strength) / 2

                        # Battery state flexibility
                        soc_flexibility = (SOC_max - current_battery_soc) / (SOC_max - SOC_min)

                        # Final charging intensity
                        charging_intensity = opportunity_score * soc_flexibility * 0.3  # Max 30% intensity

                        # Convert to actual charging power
                        max_strategic_charge = min(remaining_battery_capacity, remaining_grid_buy_capacity)
                        bes_arbitrage_charge = charging_intensity * max_strategic_charge

                        # Minimum threshold for meaningful action
                        if bes_arbitrage_charge > 0.01:
                            Pch[t] += bes_arbitrage_charge
                            Pbuy[t] += bes_arbitrage_charge / n_I

            # ========================================
            # BES-TO-GRID LOGIC
            # ========================================

            # Check for profitable BES discharge to grid
            if (Psell_max - Psell[t] > 0 and Eb[t] > Ebmin and Csell[t] > Cbw * 1.2):  # 20% profit margin minimum

                # Calculate available discharge capacity
                available_discharge_power = min(Pdch_max[t] - Pdch[t],  # Available discharge power
                    (Eb[t] - Ebmin) * ef_bat,  # Available energy
                    Psell_max - Psell[t]  # Grid selling capacity
                )

                # Check inverter capacity
                Edc2ac = Ppv[t] + Pwt[t] + Pdch[t] + Pev_dch[t] - Pch[t] - Pev_ch[t]
                available_inverter_capacity = max(0, Pinv_max - Edc2ac)
                available_discharge_power = min(available_discharge_power, available_inverter_capacity)

                if available_discharge_power > 0.01:
                    # Simple discharge decision based on price percentile
                    price_strength = max(0, min(1.0, (Csell[t] - sell_threshold_75) / max(sell_range * 0.25, 1e-6)))
                    battery_flexibility = (Eb[t] / Cn_B - SOC_min) / (SOC_max - SOC_min)

                    # Look ahead to see if better opportunities are coming (simplified)
                    future_max_sell = Csell[t]
                    for tau in range(t + 1, min(NT, t + 24)):  # Next 24 hours
                        if Csell[tau] > future_max_sell:
                            future_max_sell = Csell[tau]

                    future_opportunity_premium = (future_max_sell - Csell[t]) / max(Csell[t], 1e-6)
                    should_wait_for_better = future_opportunity_premium > 0.1  # Wait if 10%+ better price expected

                    if not should_wait_for_better:
                        discharge_intensity = price_strength * battery_flexibility * 0.3  # Max 30% intensity

                        actual_discharge = discharge_intensity * available_discharge_power

                        if actual_discharge > 0.01:
                            Pdch[t] += actual_discharge / ef_bat
                            Psell[t] += actual_discharge

            # V2G logic
            if (Psell_max - Psell[t] > 0 and Eev_dch[t] - Pev_dch[t] > 0 and Csell[t] > Cevw and (Eev[t] / C_ev > SOC_dep)):

                # 1. EV departure and lookahead analysis
                t_dep = -1
                for future_t in range(t + 1, NT):
                    if EV_p[future_t] == 0:
                        t_dep = future_t
                        break

                if t_dep == -1 or t_dep <= t:
                    optimal_lookahead = min(24, NT - t)
                    t_dep = min(t + 24, NT - 1)
                else:
                    optimal_lookahead = t_dep - t
                    for return_t in range(t_dep, min(NT, t_dep + 48)):
                        if EV_p[return_t] == 1:
                            optimal_lookahead = return_t - t
                            break

                # 2. DYNAMIC SAFETY BUFFER from historical EV usage patterns
                safety_buffer = 0
                if t >= 24:
                    # Analyze actual EV energy usage patterns
                    ev_departure_data = []
                    for hist_t in range(max(0, t - 168), t):
                        if (hist_t < len(EV_p) - 1 and hist_t < len(Eev) and
                                EV_p[hist_t] == 1 and EV_p[hist_t + 1] == 0):
                            actual_energy_at_departure = Eev[hist_t]
                            required_energy = SOC_dep * C_ev
                            buffer_used = actual_energy_at_departure - required_energy
                            ev_departure_data.append(buffer_used)

                    if len(ev_departure_data) >= 3:
                        # Manual calculation for Numba compatibility
                        n_data = len(ev_departure_data)
                        buffer_sum = 0.0
                        for i in range(n_data):
                            buffer_sum += ev_departure_data[i]
                        buffer_mean = buffer_sum / n_data

                        # Calculate variance manually
                        variance_sum = 0.0
                        for i in range(n_data):
                            variance_sum += (ev_departure_data[i] - buffer_mean) ** 2
                        buffer_std = (variance_sum / n_data) ** 0.5

                        # Safety buffer = mean + 2*std (covers ~95% of cases)
                        safety_buffer = max(0, buffer_mean + 2 * buffer_std)
                    else:
                        # Fallback: manual calculation for load variability
                        start_idx = max(0, t - 24)
                        end_idx = t
                        n_loads = end_idx - start_idx

                        if n_loads > 0:
                            load_sum = 0.0
                            for i in range(start_idx, end_idx):
                                load_sum += Eload[i]
                            load_mean = load_sum / n_loads

                            variance_sum = 0.0
                            for i in range(start_idx, end_idx):
                                variance_sum += (Eload[i] - load_mean) ** 2
                            load_std = (variance_sum / n_loads) ** 0.5
                            load_variability = load_std / max(load_mean, 1e-6)
                            safety_buffer = load_variability * (SOCe_max - SOC_dep) * C_ev
                        else:
                            safety_buffer = 0.1 * (SOCe_max - SOC_dep) * C_ev
                else:
                    # Early simulation: manual calculation for load variability
                    load_sum = 0.0
                    for i in range(NT):
                        load_sum += Eload[i]
                    load_mean = load_sum / NT

                    variance_sum = 0.0
                    for i in range(NT):
                        variance_sum += (Eload[i] - load_mean) ** 2
                    load_std = (variance_sum / NT) ** 0.5
                    load_cv = load_std / max(load_mean, 1e-6)
                    safety_buffer = load_cv * (SOCe_max - SOC_dep) * C_ev

                # 3. ECONOMIC ANALYSIS from price history
                price_history_window = min(t + 1, max(24, int(NT * 0.1)))  # At least 24h or 10% of simulation
                start_idx = max(0, t - price_history_window + 1)
                end_idx = t + 1

                # Manual price statistics calculation for Numba compatibility
                n_prices = end_idx - start_idx

                # Calculate price statistics manually
                sell_sum = 0.0
                for i in range(start_idx, end_idx):
                    sell_sum += Csell[i]
                sell_mean_local = sell_sum / n_prices

                sell_variance_sum = 0.0
                for i in range(start_idx, end_idx):
                    sell_variance_sum += (Csell[i] - sell_mean_local) ** 2
                sell_std = (sell_variance_sum / n_prices) ** 0.5

                # Create sorted price array for percentiles (manual sort)
                sell_prices_sorted = np.zeros(n_prices)
                for i in range(n_prices):
                    sell_prices_sorted[i] = Csell[start_idx + i]

                # Calculate percentile thresholds from actual price distribution
                conservative_threshold_idx = max(0, int(n_prices * 0.6))  # 60th percentile
                moderate_threshold_idx = max(0, int(n_prices * 0.75))  # 75th percentile
                aggressive_threshold_idx = max(0, int(n_prices * 0.9))  # 90th percentile

                conservative_price_threshold = sell_prices_sorted[conservative_threshold_idx]
                moderate_price_threshold = sell_prices_sorted[moderate_threshold_idx]
                aggressive_price_threshold = sell_prices_sorted[aggressive_threshold_idx]

                # Market characteristics
                price_range = sell_prices_sorted[n_prices - 1] - sell_prices_sorted[0]
                price_mean = sell_mean_local
                price_volatility = sell_std / max(price_mean, 1e-6)

                # Economic thresholds based on system costs and market efficiency
                total_system_efficiency = n_e * n_e * n_I  # Round-trip + inverter
                wear_cost_significance = Cevw / max(price_mean, 1e-6)

                # Minimum profit requirement scales with wear cost importance and market volatility
                base_profit_requirement = wear_cost_significance * (1 + price_volatility)

                # 4. SYSTEM CONSTRAINTS AND CURRENT STATE
                current_soc = Eev[t] / C_ev
                max_dischargeable = max(0, Eev[t] - SOC_dep * C_ev - safety_buffer)

                # Available discharge power
                Pb_ex = min(Eev_dch[t] - Pev_dch[t], max_dischargeable)

                # Inverter capacity constraint
                Edc2ac = Ppv[t] + Pwt[t] + Pdch[t] + Pev_dch[t] - Pch[t] - Pev_ch[t]
                available_inverter_capacity = max(0, Pinv_max - Edc2ac)
                Pb_ex = min(Pb_ex, available_inverter_capacity)

                # 5. SOLAR COMPETITION ANALYSIS
                current_solar_surplus = max(0, Ppv[t] + Pwt[t] - (Eload[t] / n_I))

                # Economic impact of solar displacement
                if current_solar_surplus > 0 and Pb_ex > 0:
                    solar_displacement_fraction = min(1.0, current_solar_surplus / Pb_ex)
                    # Cost of displacing direct solar sales (loss of inverter efficiency)
                    solar_displacement_cost = solar_displacement_fraction * (1 - n_I) * Csell[t]
                    solar_displacement_penalty_ratio = solar_displacement_cost / max(Csell[t], 1e-6)
                else:
                    solar_displacement_penalty_ratio = 0

                # 6. FUTURE OPPORTUNITY ASSESSMENT
                future_start = t + 1
                future_end = min(NT, t + optimal_lookahead)

                # Find best future selling opportunity manually
                max_future_sell = Csell[t]
                min_future_buy = Cbuy[t]

                if future_start < future_end:
                    for tau in range(future_start, future_end):
                        if tau < NT:
                            if Csell[tau] > max_future_sell:
                                max_future_sell = Csell[tau]
                            if Cbuy[tau] < min_future_buy:
                                min_future_buy = Cbuy[tau]

                future_opportunity_premium = (max_future_sell - Csell[t]) / max(Csell[t], 1e-6)
                arbitrage_potential = (Csell[t] * total_system_efficiency - min_future_buy) / max(min_future_buy, 1e-6)

                # 7. TIME FLEXIBILITY ANALYSIS
                time_until_departure = max(1, t_dep - t)

                # Minimum time needed for meaningful V2G strategy (based on system ramp rates)
                min_strategy_time = max(1, int(max_dischargeable / max(Pev_max,
                                                                       1e-6)))  # Time to discharge available energy

                # Time flexibility ratio
                time_flexibility_ratio = max(0, (time_until_departure - min_strategy_time) / max(time_until_departure, 1))

                # 8. PROFIT ANALYSIS
                gross_profit_per_unit = Csell[t] - Cevw
                net_profit_per_unit = gross_profit_per_unit - (solar_displacement_penalty_ratio * Csell[t])
                profit_margin_ratio = net_profit_per_unit / max(Cevw, 1e-6)

                # ADAPTIVE STRATEGY SELECTION - All thresholds data-driven

                do_v2g = False
                v2g_amount = 0
                #strategy_used = ""

                # STRATEGY 1: Aggressive high-value discharge (top tier prices)
                if (Csell[t] >= aggressive_price_threshold and
                        profit_margin_ratio > base_profit_requirement * 2 and  # Double the base requirement
                        future_opportunity_premium < price_volatility):  # No significantly better future (scaled by volatility)

                    do_v2g = True

                    # Discharge intensity based on how exceptional the current opportunity is
                    price_percentile_in_range = (Csell[t] - aggressive_price_threshold) / max(price_range, 1e-6)
                    profit_strength = min(2.0, profit_margin_ratio / base_profit_requirement)

                    # Combine opportunity strength with system constraints
                    base_intensity = min(1.0, price_percentile_in_range + profit_strength / 4)

                    # Reduce for solar competition, but less so for exceptional opportunities
                    solar_penalty_factor = solar_displacement_penalty_ratio * (
                            2 - profit_strength)  # Less penalty for high profits
                    final_intensity = base_intensity * (1 - solar_penalty_factor)

                    # Ensure meaningful minimum and practical maximum
                    min_meaningful_action = 0.01  # 1% of available capacity
                    max_safe_discharge = min(1.0, (current_soc - SOC_dep) * C_ev / max(max_dischargeable, 1e-6))

                    discharge_intensity = max(min_meaningful_action, min(max_safe_discharge, final_intensity))
                    v2g_amount = discharge_intensity * max_dischargeable
                    #strategy_used = "aggressive_peak"

                # STRATEGY 2: Moderate opportunity capture (mid-tier prices)
                elif (Csell[t] >= moderate_price_threshold and
                      profit_margin_ratio > base_profit_requirement and
                      time_flexibility_ratio > 0.2):  # Need some time flexibility (20% of departure time)

                    do_v2g = True

                    # Calculate intensity based on price position and profit adequacy
                    price_strength = (Csell[t] - moderate_price_threshold) / max(
                        aggressive_price_threshold - moderate_price_threshold, 1e-6)
                    profit_adequacy = profit_margin_ratio / base_profit_requirement

                    # Combine with time flexibility
                    base_intensity = min(0.8, (price_strength + profit_adequacy + time_flexibility_ratio) / 3)

                    # Apply solar penalty (higher penalty for moderate profits)
                    penalty_multiplier = 2 - profit_adequacy  # Higher penalty when profit is just adequate
                    solar_penalty = solar_displacement_penalty_ratio * penalty_multiplier
                    final_intensity = base_intensity * max(0, 1 - solar_penalty)

                    # Conservative bounds for moderate strategy
                    min_action = max_dischargeable * 0.05 / max(max_dischargeable, 1e-6)  # 5% minimum
                    max_action = max_dischargeable * 0.6 / max(max_dischargeable, 1e-6)  # 60% maximum

                    if final_intensity > min_action:
                        discharge_intensity = min(max_action, final_intensity)
                        v2g_amount = discharge_intensity * max_dischargeable
                        #strategy_used = "moderate_opportunity"

                # STRATEGY 3: Conservative arbitrage (reasonable prices with future opportunities)
                elif (Csell[t] >= conservative_price_threshold and
                      profit_margin_ratio > base_profit_requirement * 0.7 and  # 70% of base requirement
                      arbitrage_potential > base_profit_requirement and
                      time_flexibility_ratio > 0.4):  # Need good time flexibility

                    do_v2g = True

                    # Conservative approach - focus on certain opportunities
                    arbitrage_strength = min(1.0, arbitrage_potential / base_profit_requirement)
                    current_profit_strength = profit_margin_ratio / base_profit_requirement

                    # Weight current opportunity vs future arbitrage potential
                    opportunity_balance = (current_profit_strength + arbitrage_strength) / 2
                    base_intensity = min(0.5, opportunity_balance * time_flexibility_ratio)

                    # Conservative solar penalty (arbitrage strategies should avoid solar competition)
                    final_intensity = base_intensity * max(0.2, 1 - solar_displacement_penalty_ratio * 1.5)

                    # Very conservative bounds
                    min_action = max_dischargeable * 0.02 / max(max_dischargeable, 1e-6)  # 2% minimum
                    max_action = max_dischargeable * 0.4 / max(max_dischargeable, 1e-6)  # 40% maximum

                    if final_intensity > min_action:
                        discharge_intensity = min(max_action, final_intensity)
                        v2g_amount = discharge_intensity * max_dischargeable
                        #strategy_used = "conservative_arbitrage"

                # EXECUTION with final economic validation
                if do_v2g and v2g_amount > 0.001:  # Minimum economic threshold (0.1% of capacity)

                    # Convert to AC power and check grid capacity
                    available_sell_capacity = Psell_max - Psell[t]
                    v2g_power_ac = min(v2g_amount * n_I, available_sell_capacity)
                    v2g_power_dc = v2g_power_ac / n_I

                    # Final economic validation: ensure net positive after ALL costs
                    gross_revenue = v2g_power_ac * Csell[t]
                    wear_cost = v2g_power_dc * Cevw
                    solar_opportunity_cost = solar_displacement_penalty_ratio * gross_revenue
                    total_cost = wear_cost + solar_opportunity_cost

                    net_profit = gross_revenue - total_cost

                    if net_profit > 0:
                        Pev_dch[t] += v2g_power_dc
                        Psell[t] += v2g_power_ac

            Edef_DC = P_RE[t] + Pdch[t] - Pch[t] + Pev_dch[t] - Pev_ch[t]
            Esur = Eload[t] + Psell[t] - Pbuy[t] - Pdg[t] - (n_I * Edef_DC * (Edef_DC > 0)) - ((Edef_DC / n_I) * (Edef_DC < 0))
            Ens[t] = Esur * (Esur > 0)
            Edump[t] = -Esur * (Esur < 0)

            if Ens[t] > Eload[t]:
                Pac2dc = P_RE[t] + Pdg[t] + Pbuy[t] - Psell[t]
                Pev_ch[t] = (Pac2dc * n_I) + Pdch[t] - Pch[t]
                Ens[t] = Eload[t]
                if Grid > 0:
                    Pbuy[t] = max(0, min(Pbuy_max, Pbuy[t] + Ens[t]))
                    Ens[t] = max(0, Ens[t] - Pbuy_max)

        # Battery charging and discharging energy is determined based on charging and discharging power and the battery charge level is updated.
        Ech[t] = Pch[t] * dt
        Edch[t] = Pdch[t] * dt
        Eb[t + 1] = ((1 - self_discharge_rate) * Eb[t]) + (ef_bat * Ech[t]) - (Edch[t] / ef_bat)

        Eev_ch[t] = Pev_ch[t] * dt
        Eev_dch[t] = Pev_dch[t] * dt
        Eev[t + 1] = ((1 - self_discharge_rate_ev) * Eev[t]) + (n_e * Eev_ch[t]) - (Eev_dch[t] / n_e)

        if EV_p[t] == 0 and EV_p[t + 1] == 1:
            Eev[t + 1] = max(0, Eev[tdep] - (SOC_dep - SOC_arr) * C_ev)

    return Pdg, Ens, Pbuy, Psell, Edump, Pch, Pdch, Eb, Pdch_max, Pch_max, Pinv, Pev_ch, Pev_dch, Eev, Ech_req, EV_dem