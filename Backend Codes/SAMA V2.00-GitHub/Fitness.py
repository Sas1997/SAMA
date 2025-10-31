from Input_Data import InData
import numpy as np
from numba import jit
from math import ceil
from EMS import EMS
from EMS_EV import EMS_EV


Eload = InData.Eload
Ppv_r = InData.Ppv_r
Pwt_r = InData.Pwt_r
Cbt_r = InData.Cbt_r
Cdg_r = InData.Cdg_r
T = InData.T
Tc_noct = InData.Tc_noct
G = InData.G
c2 = InData.c2
fpv = InData.fpv
Gref = InData.Gref
Tcof = InData.Tcof
Tref = InData.Tref
Ta_noct = InData.Ta_noct
G_noct = InData.G_noct
n_PV =  InData.n_PV
gama =  InData.gama
Vw = InData.Vw
h_hub = InData.h_hub
h0 = InData.h0
alfa_wind_turbine = InData.alfa_wind_turbine
v_cut_in = InData.v_cut_in
v_cut_out = InData.v_cut_out
v_rated = InData.v_rated
R_B = InData.R_B
Q_lifetime_leadacid = InData.Q_lifetime_leadacid
ef_bat_leadacid = InData.ef_bat_leadacid
b = InData.b
C_fuel = InData.C_fuel
R_DG = InData.R_DG
TL_DG = InData.TL_DG
MO_DG = InData.MO_DG
SOC_max = InData.SOC_max
SOC_min = InData.SOC_min
SOC_initial = InData.SOC_initial
n_I = InData.n_I
DC_AC_ratio=InData.DC_AC_ratio
Grid = InData.Grid
Cbuy = InData.Cbuy
a = InData.a
LR_DG = InData.LR_DG
Pbuy_max = InData.Pbuy_max
Psell_max = InData.Psell_max
self_discharge_rate = InData.self_discharge_rate
alfa_battery_leadacid = InData.alfa_battery_leadacid
c = InData.c
k_lead_acid = InData.k_lead_acid
Ich_max_leadacid = InData.Ich_max_leadacid
Vnom_leadacid = InData.Vnom_leadacid
RE_incentives = InData.RE_incentives
C_PV = InData.C_PV
C_WT = InData.C_WT
C_DG = InData.C_DG
C_B = InData.C_B
C_I=InData.C_I
C_CH=InData.C_CH
Engineering_Costs=InData.Engineering_Costs
n=InData.n
L_PV=InData.L_PV
R_PV=InData.R_PV
ir=InData.ir
L_WT=InData.L_WT
R_WT=InData.R_WT
L_B=InData.L_B
L_I=InData.L_I
R_I=InData.R_I
L_CH=InData.L_CH
R_CH=InData.R_CH
MO_PV=InData.MO_PV
MO_WT=InData.MO_WT
MO_B=InData.MO_B
MO_I=InData.MO_I
MO_CH=InData.MO_CH
RT_PV=InData.RT_PV
RT_WT=InData.RT_WT
RT_B=InData.RT_B
RT_I=InData.RT_I
RT_CH=InData.RT_CH
CO2=InData.CO2
NOx=InData.NOx
SO2=InData.SO2
E_CO2=InData.E_CO2
E_SO2=InData.E_SO2
E_NOx=InData.E_NOx
Annual_expenses=InData.Annual_expenses
Service_charge=InData.Service_charge
Csell=InData.Csell
Grid_Tax=InData.Grid_Tax
System_Tax=InData.System_Tax
EM=InData.EM
LPSP_max=InData.LPSP_max
RE_min=InData.RE_min
Budget=InData.Budget
Grid_escalation = InData.Grid_escalation
C_fuel_adj = InData.C_fuel_adj
Grid_Tax_amount = InData.Grid_Tax_amount
Grid_credit = InData.Grid_credit
NEM = InData.NEM
NEM_fee = InData.NEM_fee
Lead_acid = InData.Lead_acid
Li_ion = InData.Li_ion
Ich_max_Li_ion = InData.Ich_max_Li_ion
Idch_max_Li_ion = InData.Idch_max_Li_ion
Vnom_Li_ion = InData.Vnom_Li_ion
Cnom_Li = InData.Cnom_Li
ef_bat_Li = InData.ef_bat_Li
Q_lifetime_Li = InData.Q_lifetime_Li
alfa_battery_Li_ion = InData.alfa_battery_Li_ion
P = InData.P
Eload_eh = InData.Eload_eh
# Natural Gas and Heat Pump data
HP = InData.HP
P = InData.P
Eload_hp = InData.Eload_hp
Grid_escalation_NG = InData.Grid_escalation_NG
Annual_expenses_NG = InData.Annual_expenses_NG
Service_charge_NG = InData.Service_charge_NG
Cbuy_NG = InData.Cbuy_NG
Grid_Tax_amount_NG = InData.Grid_Tax_amount_NG
Grid_credit_NG = InData.Grid_credit_NG
Hload = InData.Hload
Cload = InData.Cload
Grid_Tax_NG = InData.Grid_Tax_NG
L_HP = InData.L_HP
R_HP = InData.R_HP
RT_HP = InData.RT_HP
MO_HP = InData.MO_HP
C_HP = InData.C_HP
Php_r = InData.Php_r
NG_Grid = InData.NG_Grid

# EV data
EV = InData.EV
self_discharge_rate_ev = InData.self_discharge_rate_ev
Tin = InData.Tin
Tout = InData.Tout
C_ev = InData.C_ev
Pev_max = InData.Pev_max
SOCe_initial = InData.SOCe_initial
SOC_dep = InData.SOC_dep
SOC_arr = InData.SOC_arr
SOCe_min = InData.SOCe_min
SOCe_max = InData.SOCe_max
n_e = InData.n_e
Q_lifetime_ev = InData.Q_lifetime_ev
EV_p = InData.EV_p
R_EVB = InData.R_EVB
L_EV = InData.L_EV
RT_EV = InData.RT_EV
Cost_EV = InData.Cost_EV
MO_EV = InData.MO_EV

cap_option = InData.cap_option
cap_size = InData.cap_size
available_roof_surface = InData.available_roof_surface
PVPanel_surface_per_rated_capacity = InData.PVPanel_surface_per_rated_capacity
generation_cap = InData.generation_cap

if HP == 1:
    from BB_HP import Heat_Pump_Model

    _, power_hp_heating, power_hp_cooling, COP_hp_heating, COP_hp_cooling, hp_model, HP_size = Heat_Pump_Model(T, P / 10, Hload, Cload)
else:
    hp_model = 'No Heat Pump'
    HP_size = 0
    MO_HP = 0
    S_HP = 0
    power_hp_heating = 0
    power_hp_cooling = 0
    COP_hp_heating = 0
    COP_hp_cooling = 0

#@jit(nopython=True, fastmath=True)
def fitness(X):
    if X.size == 1:
        X = X[0]

    NT = Eload.size  # time step numbers
    Npv = round(X[0], 1)  # PV number
    Nwt = round(X[1], 2)  # WT number
    Nbat = round(X[2])  # Battery pack number
    N_DG = round(X[3], 1)  # number of Diesel Generator
    Cn_I = round(X[4], 2)  # Inverter Capacity
    N_hp = HP_size / Php_r  # Heat Pump number

    Pn_PV = Npv * Ppv_r  # PV Total Capacity
    Pn_WT = Nwt * Pwt_r  # WT Total Capacity
    Cn_B = Nbat * Cbt_r  # Battery Total Capacity
    Pn_DG = round(N_DG * Cdg_r, 4)  # Diesel Total Capacity
    Pn_hp = N_hp * Php_r
    HP_model = hp_model

    # PV Power Calculation
    #Tc = T + (((Tc_noct - 20) / 800) * G)  # Module Temprature
    # Module Temperature
    Tc = (T + 273.15 + (Tc_noct - Ta_noct) * (G / G_noct) * (1 - ((n_PV * (1 - (Tcof / 100) * (Tref + 273.15))) / gama))) / (1 + (Tc_noct - Ta_noct) * (G / G_noct) * (((Tcof / 100) * n_PV) / gama))
    Ppv = fpv * Pn_PV * (G / Gref) * (1 + (Tcof / 100) * (Tc - 273.15 - Tref))  # output power(kw)_hourly

    # Wind turbine Power Calculation
    v1 = Vw  # hourly wind speed
    v2 = ((h_hub / h0) ** (alfa_wind_turbine)) * v1  # v1 is the speed at a reference height;v2 is the speed at a hub height h2

    Pwt = np.zeros(NT)
    true_value = np.logical_and(v_cut_in <= v2, v2 < v_rated)
    Pwt[np.logical_and(v_cut_in <= v2, v2 < v_rated)] = v2[true_value] ** 3 * (Pwt_r / (v_rated ** 3 - v_cut_in ** 3)) - (v_cut_in ** 3 / (v_rated ** 3 - v_cut_in ** 3)) * (Pwt_r)
    Pwt[np.logical_and(v_rated <= v2, v2 < v_cut_out)] = Pwt_r
    Pwt = Pwt * Nwt

    ## Energy Management

    if EV > 0:
        Pdg, Ens, Pbuy, Psell, Edump, Pch, Pdch, Eb, Pdch_max, Pch_max, Pinv, Pev_ch, Pev_dch, Eev, Ech_req, EV_dem = EMS_EV(Lead_acid, Li_ion, Ich_max_Li_ion, Idch_max_Li_ion, Vnom_Li_ion, ef_bat_Li, Q_lifetime_Li, alfa_battery_Li_ion, Ppv, Pwt, Eload, Cn_B, Nbat, Pn_DG, NT, SOC_max, SOC_min, SOC_initial, n_I, Grid, Cbuy, Csell, a, b, R_DG, TL_DG, MO_DG, Cn_I, LR_DG, C_fuel, Pbuy_max, Psell_max, R_B, Q_lifetime_leadacid, self_discharge_rate, self_discharge_rate_ev, alfa_battery_leadacid, c, k_lead_acid, Ich_max_leadacid, Vnom_leadacid, ef_bat_leadacid,
         Tin, Tout, C_ev, Pev_max, SOCe_initial, SOC_dep, SOC_arr, SOCe_min, SOCe_max, n_e, Q_lifetime_ev, EV_p, R_EVB)

    else:
        Pev_ch = 0
        Pev_dch = 0
        Eev = np.zeros(NT)
        Ech_req = 0
        EV_dem = 0
        Pdg, Ens, Pbuy, Psell, Edump, Pch, Pdch, Eb, Pdch_max, Pch_max = EMS(Lead_acid, Li_ion, Ich_max_Li_ion, Idch_max_Li_ion, Cnom_Li, Vnom_Li_ion, ef_bat_Li, Q_lifetime_Li, Ppv, alfa_battery_Li_ion, Pwt, Eload, Cn_B, Nbat, Pn_DG, NT, SOC_max, SOC_min, SOC_initial, n_I, Grid, Cbuy, a, b, R_DG, TL_DG, MO_DG, Cn_I, LR_DG, C_fuel, Pbuy_max, Psell_max, R_B, Q_lifetime_leadacid, self_discharge_rate, alfa_battery_leadacid, c, k_lead_acid, Ich_max_leadacid, Vnom_leadacid, ef_bat_leadacid)

    q = (a * Pdg + b * Pn_DG) * (Pdg > 0)  # Fuel consumption of a diesel generator

    # Renewable Generation
    P_RE = Ppv + Pwt

    ## Installation and operation cost

    # Total Investment cost ($)
    I_Cost = C_PV * (1 - RE_incentives) * Pn_PV + C_WT * (1 - RE_incentives) * Pn_WT + C_DG * Pn_DG + C_B * (1 - RE_incentives) * Cn_B + C_I * (1 - RE_incentives) * Cn_I + C_CH * (1 - RE_incentives)*(Nbat > 0) + Engineering_Costs * (1 - RE_incentives) * Pn_PV + NEM_fee + (C_HP * N_hp) + Cost_EV * (EV > 0)

    Top_DG = np.sum(Pdg > 0)
    L_DG = TL_DG / max(Top_DG, 1)
    RT_DG = ceil(n / L_DG) - 1

    # Total Replacement Cost ($/year)
    R_Cost = np.zeros(n)
    # Define a resolution factor, for example 10 for deciles of a year
    res = 10
    # Multiply all times by the resolution factor
    n_res = n * res
    L_PV_res = np.int_(L_PV * res)
    L_WT_res = np.int_(L_WT * res)
    L_DG_res = np.int_(L_DG * res)
    L_B_res = np.int_(L_B * res)
    L_I_res = np.int_(L_I * res)
    L_CH_res = np.int_(L_CH * res)
    L_HP_res = np.int_(L_HP * res)
    L_EV_res = np.int_(L_EV * res)

    # Initialize arrays
    RC_PV = np.zeros(n_res)
    RC_WT = np.zeros(n_res)
    RC_DG = np.zeros(n_res)
    RC_B = np.zeros(n_res)
    RC_I = np.zeros(n_res)
    RC_CH = np.zeros(n_res)
    RC_HP = np.zeros(n_res)
    RC_EV = np.zeros(n_res)

    # Calculate replacement costs
    RC_PV[np.arange(L_PV_res, n_res, L_PV_res)] = R_PV * Pn_PV / np.power((1 + ir), 1.001 * np.arange(L_PV_res, n_res, L_PV_res) / res)
    RC_WT[np.arange(L_WT_res, n_res, L_WT_res)] = R_WT * Pn_WT / np.power((1 + ir), 1.001 * np.arange(L_WT_res, n_res, L_WT_res) / res)
    RC_DG[np.arange(L_DG_res, n_res, L_DG_res)] = R_DG * Pn_DG / np.power((1 + ir), 1.001 * np.arange(L_DG_res, n_res, L_DG_res) / res)
    RC_B[np.arange(L_B_res, n_res, L_B_res)] = R_B * Cn_B / np.power((1 + ir), 1.001 * np.arange(L_B_res, n_res, L_B_res) / res)
    RC_I[np.arange(L_I_res, n_res, L_I_res)] = R_I * Cn_I / np.power((1 + ir), 1.001 * np.arange(L_I_res, n_res, L_I_res) / res)
    RC_CH[np.arange(L_CH_res, n_res, L_CH_res)] = R_CH / np.power((1 + ir), 1.001 * np.arange(L_CH_res, n_res, L_CH_res) / res)
    RC_HP[np.arange(L_HP_res, n_res, L_HP_res)] = R_HP * N_hp / np.power((1 + ir), 1.001 * np.arange(L_HP_res, n_res, L_HP_res) / res)
    RC_EV[np.arange(L_EV_res, n_res, L_EV_res)] = R_EVB / np.power((1 + ir), 1.001 * np.arange(L_EV_res, n_res, L_EV_res) / res)

    R_Cost_res = RC_PV + RC_WT + RC_DG + RC_B + RC_I + RC_CH * (Nbat > 0) + RC_HP + RC_EV * (EV > 0)

    for i in range(n):
            R_Cost[i] = np.sum(R_Cost_res[i * res: (i + 1) * res])

    # Total M&O Cost ($/year)
    MO_Cost = (MO_PV * Pn_PV + MO_WT * Pn_WT + MO_DG * Pn_DG * np.sum(Pdg > 0) + MO_B * Cn_B + MO_I * Cn_I + MO_CH * (Nbat > 0) + MO_HP * (HP > 0) + MO_EV * (EV > 0)) / (1 + ir) ** np.arange(1, n + 1)

    # DG fuel Cost
    C_Fu = (np.sum(C_fuel * q)) * (((1 + C_fuel_adj) ** np.arange(1, n + 1)) / ((1 + ir) ** np.arange(1, n + 1)))

    # Salvage
    L_rem = (RT_PV + 1) * L_PV - n
    S_PV = (R_PV * Pn_PV) * L_rem / L_PV * 1 / (1 + ir) ** n  # PV
    L_rem = (RT_WT + 1) * L_WT - n
    S_WT = (R_WT * Pn_WT) * L_rem / L_WT * 1 / (1 + ir) ** n  # WT
    L_rem = (RT_DG + 1) * L_DG - n
    S_DG = (R_DG * Pn_DG) * L_rem / L_DG * 1 / (1 + ir) ** n  # DG
    L_rem = (RT_B + 1) * L_B - n
    S_B = (R_B * Cn_B) * L_rem / L_B * 1 / (1 + ir) ** n
    L_rem = (RT_I + 1) * L_I - n
    S_I = (R_I * Cn_I) * L_rem / L_I * 1 / (1 + ir) ** n
    L_rem = (RT_CH + 1) * L_CH - n
    S_CH = (R_CH) * L_rem / L_CH * 1 / (1 + ir) ** n
    L_rem = (RT_HP + 1) * L_HP - n
    S_HP = (R_HP * N_hp) * L_rem / L_HP * 1 / (1 + ir) ** n  # HP
    L_rem = (RT_EV + 1) * L_EV - n
    S_EV = (R_EVB) * L_rem / L_EV * 1 / (1 + ir) ** n  # EV
    Salvage = S_PV + S_WT + S_DG + S_B + S_I + S_CH * (Nbat > 0) + S_HP + S_EV * (EV > 0)

    # Emissions produced by Disesl generator (g)
    DG_Emissions = np.sum(q * (CO2 + NOx + SO2)) / 1000  # total emissions (kg/year)
    Grid_Emissions = np.sum(Pbuy * (E_CO2 + E_SO2 + E_NOx)) / 1000  # total emissions (kg/year)

    cumulative_escalation = np.cumprod(1 + Grid_escalation)
    cumulative_escalation_NG = np.cumprod(1 + Grid_escalation_NG)
    Pbuy_NG = 0  # for now until TEMS is finalized
    Pbuy_eH = Psell + Pbuy + Pdch * n_I + Edump - P_RE - Pdg - Pch * n_I - Ens - Eload_eh - power_hp_cooling # - Pev_ch  # Electrical portion for providing heating
    Pbuy_eH = np.where(Pbuy_eH > 0, Pbuy_eH, 0) * (HP > 0)
    Pbuy_C = Psell + Pbuy + Pdch * n_I + Edump - P_RE - Pdg - Pch * n_I - Ens - Eload_eh - power_hp_heating # - Pev_ch
    Pbuy_C = np.where(Pbuy_C > 0, Pbuy_C, 0) * (HP > 0)
    Pbuy_HP = Pbuy_eH + Pbuy_C
    Pbuy_p = Pbuy - Pbuy_HP

    # Total grid costs and earning in $
    # Compute Sold Electricity
    Sold_electricity = (np.sum(Psell * Csell) * (cumulative_escalation / ((1 + ir) ** np.arange(1, n + 1)))) * (Grid > 0)

    # Compute Total Grid Credits
    Total_grid_credits = (Grid_credit * (cumulative_escalation / ((1 + ir) ** np.arange(1, n + 1)))) * (Grid > 0)

    # Compute Bought Electricity for the property load (rather than HP)
    Bought_electricity_p = ((Annual_expenses + np.sum(Service_charge) + np.sum(Pbuy_p * Cbuy) + Grid_Tax_amount * np.sum(Pbuy_p)) * (cumulative_escalation / ((1 + ir) ** np.arange(1, n + 1)))) * (1 + Grid_Tax) * (Grid > 0)

    Grid_Cost_p = Bought_electricity_p
    eHeating_Cost = ((((np.sum(Pbuy_eH * Cbuy) + Grid_Tax_amount * np.sum(Pbuy_eH)) * (1 + Grid_Tax)) * (cumulative_escalation / ((1 + ir) ** np.arange(1, n + 1)))))
    gHeating_Cost = (((Annual_expenses_NG + np.sum(Service_charge_NG) + np.sum(Pbuy_NG * Cbuy_NG) + Grid_Tax_amount_NG * np.sum(Pbuy_NG)) * (cumulative_escalation_NG / ((1 + ir) ** np.arange(1, n + 1)))) * (1 + Grid_Tax_NG) - ((Grid_credit_NG) * (cumulative_escalation_NG / ((1 + ir) ** np.arange(1, n + 1))))) * (NG_Grid > 0)
    Cooling_Cost = (((np.sum(Pbuy_C * Cbuy) + Grid_Tax_amount * np.sum(Pbuy_C)) * (cumulative_escalation / ((1 + ir) ** np.arange(1, n + 1)))) * (1 + Grid_Tax))
    Grid_Cost_HP = eHeating_Cost + Cooling_Cost

    Grid_Cost_net = Grid_Cost_p + Grid_Cost_HP - Sold_electricity - Total_grid_credits

    # Capital recovery factor
    CRF = (ir * (1 + ir) ** n / ((1 + ir) ** n - 1)) if (ir != 0 and not np.isnan(ir)) else (1 / n)

    # Total Cost
    NPC = (((I_Cost + np.sum(R_Cost) + np.sum(MO_Cost) + np.sum(C_Fu) - np.sum(Salvage)) * (1 + System_Tax)) + np.sum(Grid_Cost_net) + np.sum(gHeating_Cost))
    Operating_Cost = CRF * (((np.sum(R_Cost) + np.sum(MO_Cost) + np.sum(C_Fu) - np.sum(Salvage)) * (1 + System_Tax)) + np.sum(Grid_Cost_net))

    E_tot = np.sum(Eload - Ens + Psell + Pev_ch / n_I - Pev_dch * n_I)
    E_tot = max(E_tot, 1)  # Ensures E_tot is not less than 1
    LCOE = CRF * NPC / E_tot  # Levelized Cost of Energy ($/kWh)
    LEM = (DG_Emissions + Grid_Emissions) / np.sum(Eload - Ens + Psell + Pev_ch - Pev_dch)  # Levelized Emissions (kg/kWh)

    Ebmin = SOC_min * Cn_B
    Pb_min = (Eb[1:8761] - Ebmin) + Pdch
    Ptot = (Ppv + Pwt + Pb_min) * n_I + Pdg + Grid * Pbuy_max
    DE = np.maximum(Eload - Ptot, 0)

    # Define time range
    t = np.arange(0, NT)  # Python's range includes start but excludes stop, so add 1

    # Access Eev for the specified range
    Eev2 = Eev[t]  # Adjust for 0-based indexing in Python
    # Define conditions
    ind1 = (EV_p[t - 1] == 1) & (EV_p[t] == 0)  # Adjust for 0-based indexing
    ind2 = (Eev[t] / C_ev < SOC_dep)
    ind = (ind1 == 1) & (ind2 == 1)

    # Calculate ENS of EV charge
    Ens_EV = np.roll(ind * (SOC_dep * C_ev - Eev[t]), -1) * (EV > 0)
    Ens_EV[-1] = 0

    # Calculate LPSP
    LPSP = (np.sum(Ens) + np.sum(Ens_EV)) / (np.sum(Eload) + np.sum(EV_dem))
    # Calculate Pcuns_dc
    Pcuns_dc = Pev_ch + Pch - Pev_dch - Pdch

    # Calculate RE
    RE = 1 - np.sum(Pdg + Pbuy) / np.sum(Eload + Psell - Ens + Pcuns_dc * (Pcuns_dc > 0) / n_I + Pcuns_dc * (Pcuns_dc < 0) * n_I)

    if np.isnan(RE):
        RE = 0

    def smooth_penalty(violation, threshold=0, scale=1.0, smoothness=2):
        """
        Smooth penalty function that provides gradients for optimization

        Parameters:
        - violation: constraint violation amount
        - threshold: violation threshold
        - scale: penalty scaling factor
        - smoothness: controls penalty curve smoothness (higher = smoother)

        Returns:
        - penalty: smooth penalty value
        """
        if violation <= threshold:
            return 0.0
        else:
            # Smooth exponential penalty
            normalized_violation = (violation - threshold) / (abs(threshold) + 1e-6)
            return scale * (normalized_violation ** smoothness)


    # Normalize objective components to similar scales
    # Scale factors (adjust based on your typical values)
    NPC_scale = 1e-5  # Assuming NPC is in range 1e6-1e7
    EM_scale = 1e2  # Assuming EM is in range 0.001-0.1

    # Main objective (normalized)
    objective = NPC_scale * NPC + EM_scale * EM * LEM

    # Constraint violation measures with smooth penalties
    violations = 0

    # 1. Grid cost constraint (Net Energy Metering)
    if NEM == 1:
        grid_cost_violation = max(0, -np.sum(Grid_Cost_net))
        violations += smooth_penalty(grid_cost_violation, threshold=0, scale=1e-4)

    # 2. Roof surface constraint
    if cap_option == 3:
        surface_violation = max(0, Npv * PVPanel_surface_per_rated_capacity - available_roof_surface)
        violations += smooth_penalty(surface_violation, threshold=0, scale=1e-5)

    # 3. PV capacity constraint
    if cap_option == 1 and NEM == 1:
        pv_cap_violation = max(0, Pn_PV - cap_size)
        violations += smooth_penalty(pv_cap_violation, threshold=0, scale=1e-5)

    # 4. Generation capacity constraint
    if cap_option == 2 and NEM == 1:
        total_generation = np.sum(Ppv + Pwt)
        max_allowed_generation = (generation_cap / 100) * np.sum(Eload + EV_dem)
        gen_violation = max(0, total_generation - max_allowed_generation)
        violations += smooth_penalty(gen_violation, threshold=0, scale=1e-6)

    # 5. Energy dump penalty (minimize waste)
    if NEM == 1:
        dump_penalty = np.sum(Edump)
        violations += 1e-4 * dump_penalty  # Linear penalty for energy waste

    # 6. LPSP constraint (reliability)
    lpsp_violation = max(0, LPSP - LPSP_max)
    violations += smooth_penalty(lpsp_violation, threshold=0, scale=1e2)

    # 7. Renewable energy constraint
    re_violation = max(0, RE_min - RE)
    violations += smooth_penalty(re_violation, threshold=0, scale=1e1)

    # 8. Budget constraint
    budget_violation = max(0, I_Cost - Budget)
    violations += smooth_penalty(budget_violation, threshold=0, scale=1e-6)

    # Total fitness with balanced scaling
    Z = objective + violations


    return Z



