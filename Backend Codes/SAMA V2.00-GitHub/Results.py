from Input_Data import InData
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import pandas as pd
from math import ceil
from EMS import EMS
from EMS_EV import EMS_EV
from EV_travel import compute_ev_travel_energy
from EV_travel import calculate_energy_consumption
import numpy_financial as npf
import csv
import matplotlib.ticker as mticker

# Loading all inputs
WT=InData.WT
daysInMonth = InData.daysInMonth
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
Cash_Flow_adv = InData.Cash_Flow_adv
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
def Gen_Results(X):
    if (len(X)) == 1:
        X = X[0]

    NT = Eload.size  # time step numbers
    Npv = round(X[0], 1)  # PV number
    Nwt = round(X[1], 2)  # WT number
    Nbat = round(X[2])  # Battery pack number
    N_DG = round(X[3], 1)  # number of Diesel Generator
    Cn_I = round(X[4], 2)  # Inverter Capacity
    Cn_I_mg = Cn_I
    N_hp = HP_size / Php_r # Heat Pump number

    Pn_PV = Npv * Ppv_r  # PV Total Capacity
    Pn_WT = Nwt * Pwt_r  # WT Total Capacity
    Cn_B = Nbat * Cbt_r  # Battery Total Capacity
    Pn_DG = round(N_DG * Cdg_r, 4)  # Diesel Total Capacity
    Pn_hp = N_hp * Php_r
    HP_model = hp_model

    # if (Pn_PV >= DC_AC_ratio * (Cn_I + Pn_WT + Pn_DG + Pbuy_max * (Grid > 0))):
    #     Cn_I = round(Pn_PV / DC_AC_ratio - Pn_DG, 2)


    # PV Power Calculation
    #Tc = T + (((Tc_noct - 20) / 800) * G)  # Module Temprature
    Tc = (T + 273.15 + (Tc_noct - Ta_noct) * (G / G_noct) * (1 - ((n_PV * (1 - (Tcof / 100) * (Tref + 273.15))) / gama))) / (1 + (Tc_noct - Ta_noct) * (G / G_noct) * (((Tcof / 100) * n_PV) / gama))
    Ppv = fpv * Pn_PV * (G / Gref) * (1 + (Tcof / 100) * (Tc - 273.15 - Tref))  # output power(kw)_hourly

    # Wind turbine Power Calculation
    v1 = Vw  # hourly wind speed
    v2 = ((h_hub / h0) ** (alfa_wind_turbine)) * v1  # v1 is the speed at a reference height;v2 is the speed at a hub height h2

    Pwt = np.zeros(8760)
    true_value = np.logical_and(v_cut_in <= v2, v2 < v_rated)
    Pwt[np.logical_and(v_cut_in <= v2, v2 < v_rated)] = v2[true_value] ** 3 * (Pwt_r / (v_rated ** 3 - v_cut_in ** 3)) - (v_cut_in ** 3 / (v_rated ** 3 - v_cut_in ** 3)) * (Pwt_r)
    Pwt[np.logical_and(v_rated <= v2, v2 < v_cut_out)] = Pwt_r
    Pwt = Pwt * Nwt

    ## Energy Management

    if EV > 0:
        Pdg, Ens, Pbuy, Psell, Edump, Pch, Pdch, Eb, Pdch_max, Pch_max, Pinv, Pev_ch, Pev_dch, Eev, Ech_req, EV_dem = EMS_EV(
            Lead_acid, Li_ion, Ich_max_Li_ion, Idch_max_Li_ion, Vnom_Li_ion, ef_bat_Li, Q_lifetime_Li,
            alfa_battery_Li_ion, Ppv, Pwt, Eload, Cn_B, Nbat, Pn_DG, NT, SOC_max, SOC_min, SOC_initial, n_I, Grid,
            Cbuy, Csell, a, b, R_DG, TL_DG, MO_DG, Cn_I, LR_DG, C_fuel, Pbuy_max, Psell_max, R_B,
            Q_lifetime_leadacid, self_discharge_rate, self_discharge_rate_ev, alfa_battery_leadacid, c, k_lead_acid,
            Ich_max_leadacid, Vnom_leadacid, ef_bat_leadacid,
            Tin, Tout, C_ev, Pev_max, SOCe_initial, SOC_dep, SOC_arr, SOCe_min, SOCe_max, n_e, Q_lifetime_ev, EV_p,
            R_EVB)

    else:
        Pev_ch = 0
        Pev_dch = 0
        Eev = np.zeros(NT + 1)
        Ech_req = 0
        EV_dem = 0
        Pdg, Ens, Pbuy, Psell, Edump, Pch, Pdch, Eb, Pdch_max, Pch_max = EMS(Lead_acid, Li_ion, Ich_max_Li_ion,
                                                                             Idch_max_Li_ion, Cnom_Li, Vnom_Li_ion,
                                                                             ef_bat_Li, Q_lifetime_Li, Ppv,
                                                                             alfa_battery_Li_ion, Pwt, Eload, Cn_B,
                                                                             Nbat, Pn_DG, NT, SOC_max, SOC_min,
                                                                             SOC_initial, n_I, Grid, Cbuy, a, b,
                                                                             R_DG, TL_DG, MO_DG, Cn_I, LR_DG,
                                                                             C_fuel, Pbuy_max, Psell_max, R_B,
                                                                             Q_lifetime_leadacid,
                                                                             self_discharge_rate,
                                                                             alfa_battery_leadacid, c, k_lead_acid,
                                                                             Ich_max_leadacid, Vnom_leadacid,
                                                                             ef_bat_leadacid)


    if np.sum(Ppv) < 0.1:
        Pn_PV = 0
        if EV > 0:
            Pdg, Ens, Pbuy, Psell, Edump, Pch, Pdch, Eb, Pdch_max, Pch_max, Pinv, Pev_ch, Pev_dch, Eev, Ech_req, EV_dem = EMS_EV(
                Lead_acid, Li_ion, Ich_max_Li_ion, Idch_max_Li_ion, Vnom_Li_ion, ef_bat_Li, Q_lifetime_Li,
                alfa_battery_Li_ion, Ppv, Pwt, Eload, Cn_B, Nbat, Pn_DG, NT, SOC_max, SOC_min, SOC_initial, n_I, Grid,
                Cbuy, Csell, a, b, R_DG, TL_DG, MO_DG, Cn_I, LR_DG, C_fuel, Pbuy_max, Psell_max, R_B,
                Q_lifetime_leadacid, self_discharge_rate, self_discharge_rate_ev, alfa_battery_leadacid, c, k_lead_acid,
                Ich_max_leadacid, Vnom_leadacid, ef_bat_leadacid,
                Tin, Tout, C_ev, Pev_max, SOCe_initial, SOC_dep, SOC_arr, SOCe_min, SOCe_max, n_e, Q_lifetime_ev, EV_p, R_EVB)

        else:
            Pev_ch = 0
            Pev_dch = 0
            Eev = np.zeros(NT + 1)
            Ech_req = 0
            EV_dem = 0
            Pdg, Ens, Pbuy, Psell, Edump, Pch, Pdch, Eb, Pdch_max, Pch_max = EMS(Lead_acid, Li_ion, Ich_max_Li_ion,
                                                                                 Idch_max_Li_ion, Cnom_Li, Vnom_Li_ion,
                                                                                 ef_bat_Li, Q_lifetime_Li, Ppv,
                                                                                 alfa_battery_Li_ion, Pwt, Eload, Cn_B,
                                                                                 Nbat, Pn_DG, NT, SOC_max, SOC_min,
                                                                                 SOC_initial, n_I, Grid, Cbuy, a, b,
                                                                                 R_DG, TL_DG, MO_DG, Cn_I, LR_DG,
                                                                                 C_fuel, Pbuy_max, Psell_max, R_B,
                                                                                 Q_lifetime_leadacid,
                                                                                 self_discharge_rate,
                                                                                 alfa_battery_leadacid, c, k_lead_acid,
                                                                                 Ich_max_leadacid, Vnom_leadacid,
                                                                                 ef_bat_leadacid)

    if np.sum(Pwt) < 0.1:
        Pn_WT = 0
        if EV > 0:
            Pdg, Ens, Pbuy, Psell, Edump, Pch, Pdch, Eb, Pdch_max, Pch_max, Pinv, Pev_ch, Pev_dch, Eev, Ech_req, EV_dem = EMS_EV(
                Lead_acid, Li_ion, Ich_max_Li_ion, Idch_max_Li_ion, Vnom_Li_ion, ef_bat_Li, Q_lifetime_Li,
                alfa_battery_Li_ion, Ppv, Pwt, Eload, Cn_B, Nbat, Pn_DG, NT, SOC_max, SOC_min, SOC_initial, n_I, Grid,
                Cbuy, Csell, a, b, R_DG, TL_DG, MO_DG, Cn_I, LR_DG, C_fuel, Pbuy_max, Psell_max, R_B,
                Q_lifetime_leadacid, self_discharge_rate, self_discharge_rate_ev, alfa_battery_leadacid, c, k_lead_acid,
                Ich_max_leadacid, Vnom_leadacid, ef_bat_leadacid,
                Tin, Tout, C_ev, Pev_max, SOCe_initial, SOC_dep, SOC_arr, SOCe_min, SOCe_max, n_e, Q_lifetime_ev, EV_p, R_EVB)

        else:
            Pev_ch = 0
            Pev_dch = 0
            Eev = np.zeros(NT + 1)
            Ech_req = 0
            EV_dem = 0
            Pdg, Ens, Pbuy, Psell, Edump, Pch, Pdch, Eb, Pdch_max, Pch_max = EMS(Lead_acid, Li_ion, Ich_max_Li_ion,
                                                                                 Idch_max_Li_ion, Cnom_Li, Vnom_Li_ion,
                                                                                 ef_bat_Li, Q_lifetime_Li, Ppv,
                                                                                 alfa_battery_Li_ion, Pwt, Eload, Cn_B,
                                                                                 Nbat, Pn_DG, NT, SOC_max, SOC_min,
                                                                                 SOC_initial, n_I, Grid, Cbuy, a, b,
                                                                                 R_DG, TL_DG, MO_DG, Cn_I, LR_DG,
                                                                                 C_fuel, Pbuy_max, Psell_max, R_B,
                                                                                 Q_lifetime_leadacid,
                                                                                 self_discharge_rate,
                                                                                 alfa_battery_leadacid, c, k_lead_acid,
                                                                                 Ich_max_leadacid, Vnom_leadacid,
                                                                                 ef_bat_leadacid)

    if np.sum(Pdg) < 0.1:
        Pn_DG = 0
        if EV > 0:
            Pdg, Ens, Pbuy, Psell, Edump, Pch, Pdch, Eb, Pdch_max, Pch_max, Pinv, Pev_ch, Pev_dch, Eev, Ech_req, EV_dem = EMS_EV(
                Lead_acid, Li_ion, Ich_max_Li_ion, Idch_max_Li_ion, Vnom_Li_ion, ef_bat_Li, Q_lifetime_Li,
                alfa_battery_Li_ion, Ppv, Pwt, Eload, Cn_B, Nbat, Pn_DG, NT, SOC_max, SOC_min, SOC_initial, n_I, Grid,
                Cbuy, Csell, a, b, R_DG, TL_DG, MO_DG, Cn_I, LR_DG, C_fuel, Pbuy_max, Psell_max, R_B,
                Q_lifetime_leadacid, self_discharge_rate, self_discharge_rate_ev, alfa_battery_leadacid, c, k_lead_acid,
                Ich_max_leadacid, Vnom_leadacid, ef_bat_leadacid,
                Tin, Tout, C_ev, Pev_max, SOCe_initial, SOC_dep, SOC_arr, SOCe_min, SOCe_max, n_e, Q_lifetime_ev, EV_p, R_EVB)

        else:
            Pev_ch = 0
            Pev_dch = 0
            Eev = np.zeros(NT + 1)
            Ech_req = 0
            EV_dem = 0
            Pdg, Ens, Pbuy, Psell, Edump, Pch, Pdch, Eb, Pdch_max, Pch_max = EMS(Lead_acid, Li_ion, Ich_max_Li_ion,
                                                                                 Idch_max_Li_ion, Cnom_Li, Vnom_Li_ion,
                                                                                 ef_bat_Li, Q_lifetime_Li, Ppv,
                                                                                 alfa_battery_Li_ion, Pwt, Eload, Cn_B,
                                                                                 Nbat, Pn_DG, NT, SOC_max, SOC_min,
                                                                                 SOC_initial, n_I, Grid, Cbuy, a, b,
                                                                                 R_DG, TL_DG, MO_DG, Cn_I, LR_DG,
                                                                                 C_fuel, Pbuy_max, Psell_max, R_B,
                                                                                 Q_lifetime_leadacid,
                                                                                 self_discharge_rate,
                                                                                 alfa_battery_leadacid, c, k_lead_acid,
                                                                                 Ich_max_leadacid, Vnom_leadacid,
                                                                                 ef_bat_leadacid)

    if np.sum(Pch) < 0.1 or np.sum(Pdch) < 0.1:
        Cn_B = 0
        if EV > 0:
            Pdg, Ens, Pbuy, Psell, Edump, Pch, Pdch, Eb, Pdch_max, Pch_max, Pinv, Pev_ch, Pev_dch, Eev, Ech_req, EV_dem = EMS_EV(
                Lead_acid, Li_ion, Ich_max_Li_ion, Idch_max_Li_ion, Vnom_Li_ion, ef_bat_Li, Q_lifetime_Li,
                alfa_battery_Li_ion, Ppv, Pwt, Eload, Cn_B, Nbat, Pn_DG, NT, SOC_max, SOC_min, SOC_initial, n_I, Grid,
                Cbuy, Csell, a, b, R_DG, TL_DG, MO_DG, Cn_I, LR_DG, C_fuel, Pbuy_max, Psell_max, R_B,
                Q_lifetime_leadacid, self_discharge_rate, self_discharge_rate_ev, alfa_battery_leadacid, c, k_lead_acid,
                Ich_max_leadacid, Vnom_leadacid, ef_bat_leadacid,
                Tin, Tout, C_ev, Pev_max, SOCe_initial, SOC_dep, SOC_arr, SOCe_min, SOCe_max, n_e, Q_lifetime_ev,
                EV_p, R_EVB)

        else:
            Pev_ch = 0
            Pev_dch = 0
            Eev = np.zeros(NT + 1)
            Ech_req = 0
            EV_dem = 0
            Pdg, Ens, Pbuy, Psell, Edump, Pch, Pdch, Eb, Pdch_max, Pch_max = EMS(Lead_acid, Li_ion, Ich_max_Li_ion,
                                                                                 Idch_max_Li_ion, Cnom_Li, Vnom_Li_ion,
                                                                                 ef_bat_Li, Q_lifetime_Li, Ppv,
                                                                                 alfa_battery_Li_ion, Pwt, Eload, Cn_B,
                                                                                 Nbat, Pn_DG, NT, SOC_max, SOC_min,
                                                                                 SOC_initial, n_I, Grid, Cbuy, a, b,
                                                                                 R_DG, TL_DG, MO_DG, Cn_I, LR_DG,
                                                                                 C_fuel, Pbuy_max, Psell_max, R_B,
                                                                                 Q_lifetime_leadacid,
                                                                                 self_discharge_rate,
                                                                                 alfa_battery_leadacid, c, k_lead_acid,
                                                                                 Ich_max_leadacid, Vnom_leadacid,
                                                                                 ef_bat_leadacid)

    if Pn_PV == 0 and Pn_WT == 0 and Cn_B == 0 and C_ev == 0:
        Cn_I = 0

    Cn_I_min_required = np.max(Eload)
    Cn_I = max(Cn_I, round(Cn_I_min_required, 2))

    q = (a * Pdg + b * Pn_DG) * (Pdg > 0)  # Fuel consumption of a diesel generator

    # Define time range
    t = np.arange(0, NT)

    # Access Eev for the specified range
    Eev2 = Eev[t]  # Adjust for 0-based indexing in Python
    # Define conditions
    ind1 = (EV_p[t - 1] == 1) & (EV_p[t] == 0)  # Adjust for 0-based indexing
    ind2 = (Eev[t] / C_ev < SOC_dep)
    ind = (ind1 == 1) & (ind2 == 1)

    # Calculate ENS of EV charge
    Ens_EV = np.roll(ind * (SOC_dep * C_ev - Eev[t]), -1) * (EV > 0)
    Ens_EV[-1] = 0 #np.maximum(SOC_dep * C_ev - Eev[-1], 0)
    # Battery Power
    P_bat = Pch - Pdch

    # EV Power
    P_ev_bat = Pev_ch - Pev_dch
    # Served load
    # Renewable Generation
    P_RE = Ppv + Pwt
    Eload_served = Eload + EV_dem - Ens - Ens_EV
    Eload_served_p = Eload - Ens
    P_RE_served = pd.Series(np.select([P_RE > Eload, P_RE <= Eload], [Eload, P_RE]))

    ## Installation and operation cost

    # Total Investment cost ($)
    I_Cost = C_PV * (1 - RE_incentives) * Pn_PV + C_WT * (1 - RE_incentives) * Pn_WT + C_DG * Pn_DG + C_B * (1 - RE_incentives) * Cn_B + C_I * (1 - RE_incentives) * Cn_I + C_CH * (1 - RE_incentives) * (Cn_B > 0) + Engineering_Costs * (1 - RE_incentives) * Pn_PV + NEM_fee + (C_HP * N_hp) + Cost_EV * (EV > 0)
    I_Cost_without_incentives = C_PV * Pn_PV + C_WT * Pn_WT + C_DG * Pn_DG + C_B * Cn_B + C_I * Cn_I + C_CH*(Nbat > 0) + Engineering_Costs * Pn_PV + NEM_fee + C_HP * N_hp
    Total_incentives_received = I_Cost_without_incentives - I_Cost
    Solar_Cost_Initial = C_PV * (1 - RE_incentives) * Pn_PV + C_I * (1 - RE_incentives) * Cn_I + Engineering_Costs * (1 - RE_incentives) * Pn_PV + NEM_fee

    Top_DG = np.sum(Pdg > 0)
    L_DG = TL_DG / max(Top_DG, 1)
    RT_DG = ceil(n / L_DG) - 1

    # Total Replacement cost ($)
    R_Cost= np.zeros(n)
    Solar_Cost_replacement = np.zeros(n)
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
    RC_EV[np.arange(L_EV_res, n_res, L_EV_res)] = R_EVB / np.power((1 + ir), 1.001 * np.arange(L_EV_res, n_res,L_EV_res) / res)


    R_Cost_res = RC_PV + RC_WT + RC_DG + RC_B + RC_I + RC_CH * (Nbat > 0) + RC_HP + RC_EV * (EV > 0)
    Solar_Cost_R = RC_PV + RC_I

    for i in range(n):
            R_Cost[i] = np.sum(R_Cost_res[i * res: (i + 1) * res])
            Solar_Cost_replacement[i] = np.sum(Solar_Cost_R[i * res: (i + 1) * res])

    # Total M&O Cost ($/year)
    MO_Cost = (MO_PV * Pn_PV + MO_WT * Pn_WT + MO_DG * Pn_DG * np.sum(Pdg > 0) + MO_B * Cn_B + MO_I * Cn_I + MO_CH * (Nbat > 0) + MO_HP * (HP > 0) + MO_EV * (EV > 0)) / (1 + ir) ** np.arange(1, n + 1)
    Solar_Cost_MO = (MO_PV * Pn_PV + MO_I * Cn_I) / (1 + ir) ** np.arange(1, n + 1)

    # DG fuel Cost
    C_Fu = (np.sum(C_fuel * q)) * (((1 + C_fuel_adj) ** np.arange(1, n + 1)) / ((1 + ir) ** np.arange(1, n + 1)))

    # Salvage
    Salvage = np.zeros(n)
    Salvage_Solar = np.zeros(n)

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
    Salvage_S = S_PV + S_WT + S_DG + S_B + S_I + S_CH * (Nbat > 0) + S_HP + S_EV * (EV > 0)
    Salvage_Solar[-1] = S_PV + S_I
    Salvage[-1] = Salvage_S

    # Emissions produced by Disesl generator (g)
    DG_Emissions = np.sum(q * (CO2 + NOx + SO2)) # total emissions (kg/year)

    Grid_Emissions = np.sum(Pbuy * (E_CO2 + E_SO2 + E_NOx))  # total emissions (kg/year)
    Grid_only_Emissions = np.sum((Eload + EV_dem) * (E_CO2 + E_SO2 + E_NOx))

    # Grid_escalation = np.array ([0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02])
    cumulative_escalation = np.cumprod(1 + Grid_escalation)
    cumulative_escalation_NG = np.cumprod(1 + Grid_escalation_NG)
    Pbuy_NG = 0  # for now until TEMS is finalized
    Pbuy_eH = Psell + Pbuy + Pdch * n_I + Edump - P_RE - Pdg - Pch * n_I - Ens - Eload_eh - power_hp_cooling #- Pev_ch / 1  # Electrical portion for providing heating
    Pbuy_eH = np.where(Pbuy_eH > 0, Pbuy_eH, 0) * (HP > 0)
    Pbuy_C = Psell + Pbuy + Pdch * n_I + Edump - P_RE - Pdg - Pch * n_I - Ens - Eload_eh - power_hp_heating #- Pev_ch / 1
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
    #print(np.sum(Grid_Cost_HP))
    #print(np.sum(Grid_Cost_p))

    Grid_Cost_onlyG = (((Annual_expenses + np.sum(Service_charge) + np.sum((Eload + EV_dem) * Cbuy) + Grid_Tax_amount * np.sum((Eload + EV_dem))) * (cumulative_escalation / ((1 + ir) ** np.arange(1, n + 1)))) * (1 + Grid_Tax) - ((Grid_credit) * (cumulative_escalation / ((1 + ir) ** np.arange(1, n + 1)))))
    NG_Grid_Cost_onlyG = (((Annual_expenses_NG + np.sum(Service_charge_NG) + np.sum(Hload * Cbuy_NG) + Grid_Tax_amount_NG * np.sum(Hload)) * (cumulative_escalation_NG / ((1 + ir) ** np.arange(1, n + 1)))) * (1 + Grid_Tax_NG) - ((Grid_credit_NG) * (cumulative_escalation_NG / ((1 + ir) ** np.arange(1, n + 1))))) * (NG_Grid > 0)

    Solar_Cost = (Solar_Cost_Initial + np.sum(Solar_Cost_replacement) + np.sum(Solar_Cost_MO) - np.sum(Salvage_Solar)) * (1 + System_Tax)

    # Compute Grid Avoidable Cost
    Grid_avoidable_cost = ((np.sum((Eload + EV_dem) * Cbuy) + Grid_Tax_amount * np.sum((Eload + EV_dem))) * (cumulative_escalation / ((1 + ir) ** np.arange(1, n + 1)))) * (1 + Grid_Tax)

    # Compute Grid Unavoidable Cost
    Grid_unavoidable_cost = (((Annual_expenses + np.sum(Service_charge)) * (cumulative_escalation / ((1 + ir) ** np.arange(1, n + 1)))) * (1 + Grid_Tax) - (Grid_credit * (cumulative_escalation / ((1 + ir) ** np.arange(1, n + 1)))))

    # Capital recovery factor
    CRF = (ir * (1 + ir) ** n / ((1 + ir) ** n - 1)) if (ir != 0 and not np.isnan(ir)) else (1 / n)

    # Total Cost
    NPC = (((I_Cost + np.sum(R_Cost) + np.sum(MO_Cost) + np.sum(C_Fu) - np.sum(Salvage)) * (1 + System_Tax)) + np.sum(Grid_Cost_net) + np.sum(gHeating_Cost))
    NPC_without_incentives = (((I_Cost_without_incentives + np.sum(R_Cost) + np.sum(MO_Cost) + np.sum(C_Fu) - np.sum(Salvage)) * (1 + System_Tax)) + np.sum(Grid_Cost_net))
    NPC_Grid = np.sum(Grid_Cost_onlyG)

    Heating_Cost_total = np.sum(eHeating_Cost) + np.sum(gHeating_Cost)
    Cooling_Cost_total = np.sum(Cooling_Cost)

    Operating_Cost = CRF * (((np.sum(R_Cost) + np.sum(MO_Cost) + np.sum(C_Fu) - np.sum(Salvage)) * (1 + System_Tax)) + np.sum(Grid_Cost_net))

    if np.sum(Psell) < 0.1:
        Solar_Cost_perkWh = CRF * Solar_Cost / np.sum(Eload_served) #???
    else:
        Solar_Cost_perkWh = CRF * Solar_Cost / (np.sum(Ppv) - np.sum(Edump))

    E_tot = np.sum(Eload - Ens + Psell + Pev_ch / n_I - Pev_dch * n_I)
    E_tot = max(E_tot, 1)  # Ensures E_tot is not less than 1
    LCOE = CRF * NPC / E_tot  # Levelized Cost of Energy ($/kWh)
    LCOE_without_incentives = CRF * NPC_without_incentives / E_tot
    LCOE_Grid = CRF * NPC_Grid / np.sum(Eload)


    Grid_avoidable_cost_perkWh = CRF * np.sum(Grid_avoidable_cost) / np.sum((Eload + EV_dem))
    Grid_unavoidable_cost_perkWh = CRF * np.sum(Grid_unavoidable_cost) / np.sum((Eload + EV_dem))

    LEM = (DG_Emissions + Grid_Emissions) / np.sum(Eload - Ens + Psell + Pev_ch - Pev_dch)  # Levelized Emissions (kg/kWh)

    Ebmin = SOC_min * Cn_B  # Battery minimum energy
    Pb_min= (Eb[1:8761] - Ebmin) + Pdch  # Battery minimum power in t=2,3,...,NT
    Ptot = (Ppv + Pwt + Pb_min) * n_I + Pdg + Grid * Pbuy_max  # total available power in system for each hour
    DE = np.maximum((Eload - Ptot), 0)  # power shortage in each hour

    # Calculate LPSP
    LPSP = (np.sum(Ens) + np.sum(Ens_EV)) / (np.sum(Eload) + np.sum(EV_dem))
    LPSP_p = (np.sum(Ens)) / (np.sum(Eload))
    LPSP_ev = (np.sum(Ens_EV)) / (np.sum(EV_dem)) if EV > 0 else 0
    # Calculate Pcuns_dc
    Pcuns_dc = Pev_ch + Pch - Pev_dch - Pdch

    # Calculate RE
    RE = 1 - np.sum(Pdg + Pbuy) / np.sum(Eload + Psell - Ens + Pcuns_dc * (Pcuns_dc > 0) / n_I + Pcuns_dc * (Pcuns_dc < 0) * n_I)

    if np.isnan(RE):
        RE = 0


    # Avoided costs calc
    P_served_other_than_grid = Eload - Ens * (Eload > Ens) + Pev_ch / n_I - Pbuy

    avoided_costs_e = ((np.sum(Eload_served * Cbuy) + Annual_expenses + np.sum(Service_charge) + Grid_Tax_amount * np.sum(Eload_served)) * (cumulative_escalation / ((1 + ir) ** np.arange(1, n + 1))) * (1 + Grid_Tax)) - (Grid_credit * (cumulative_escalation / ((1 + ir) ** np.arange(1, n + 1))))
    avoided_costs_ng = (((np.sum(Hload * Cbuy_NG) + Annual_expenses_NG + np.sum(Service_charge_NG) + Grid_Tax_amount_NG * np.sum(Hload)) * (cumulative_escalation_NG / ((1 + ir) ** np.arange(1, n + 1))) * (1 + Grid_Tax_NG)) - (Grid_credit_NG * (cumulative_escalation_NG / ((1 + ir) ** np.arange(1, n + 1))))) * (NG_Grid > 0)

    Eev_travel = compute_ev_travel_energy(Eev, EV_p, SOC_dep, SOC_arr, C_ev) if EV > 0 else np.zeros(NT)
    Eev[1:8761][EV_p == 0] = Eev_travel[EV_p == 0]
    Pev_travel = calculate_energy_consumption(Eev, EV_p)
    if EV_p[0] == 0:  # If EV is traveling at t=0
        Pev_travel[0] = Eev_travel[0] - Eev_travel[1]


    # Extracting data for plotting
    data = {'Ppv': Ppv, 'Pdg': Pdg, 'Pch': Pch, 'Pch_ev':Pev_ch, 'Pdch': Pdch, 'Pdch_ev': Pev_dch, 'Pdch_max': Pdch_max, 'Pch_max': Pch_max, 'Eb': Eb[1:8761], 'Eev': Eev[1:8761], 'Eev_travel': Eev_travel, 'Pev_travel': Pev_travel, 'SOC': Eb[1:8761] / Cn_B if (Cn_B != 0 and not np.isnan(Cn_B)) else 0, 'Pbuy total':Pbuy, 'Psell':Psell, 'Pbuy_HP':Pbuy_HP, 'Pbuy_eH': Pbuy_eH, 'Pbuy_C': Pbuy_C, 'Eload':Eload, 'Eload_e':Eload_eh, 'Eload_hp': Eload_hp, 'EV_dem': EV_dem, 'Eload_hp_heating': power_hp_heating, 'Eload_hp_cooling': power_hp_cooling,'COP_hp_heating': COP_hp_heating, 'COP_hp_cooling': COP_hp_cooling, 'ENS': Ens, 'ENS_EV': Ens_EV, 'Edump': Edump, 'P_RE_served':P_RE_served, 'Csell':Csell, 'Cbuy':Cbuy, 'Pserved':P_served_other_than_grid, 'POA':G, 'Temperature':T, "Wind Speed":Vw, 'EV_p': EV_p, 'E_charge_required': Ech_req}
    df = pd.DataFrame(data)
    df.to_csv('output/data/Outputforplotting.csv', index=False)

    text = "System Size"
    line = text.center(50, "-")
    print(line)
    print('Cpv  (kW) =', Pn_PV)
    if WT == 1:
        print('Cwt  (kW) =', Pn_WT)
    print('Cbat (kWh) =', round(Cn_B, 4))
    print('Cdg  (kW) =', Pn_DG)
    print('Cinverter (kW) =', Cn_I)
    print('Inverter maximum Handeling Power (kW) =', Cn_I_mg)
    (HP == 1) and print("Selected heat pump based on the maximum thermal load: Goodman Heat Pump Model ", HP_model, " with the capacity of ", Pn_hp, " BTU/hr")

    print(' ')
    text = "Economic Results"
    line = text.center(50, "*")
    print(line)
    print('NPC  = $', round(NPC, 2))
    print('NPC without incentives = $', round(NPC_without_incentives, 2))
    print('Total Solar Cost = $', round(Solar_Cost, 2))
    print('NPC for only Grid connected system = $', round(NPC_Grid, 2))
    print('Total Heating Cost = $', round(Heating_Cost_total, 2))
    print('Total Cooling Cost = $', round(Cooling_Cost_total, 2))
    print('Total Heat Pump Cost = $', round(np.sum(Grid_Cost_HP), 2))
    print('NPC for only Natural Gas Grid connection = $', round(np.sum(NG_Grid_Cost_onlyG), 2))
    print('Total Grid avoidable cost = $', round(np.sum(Grid_avoidable_cost), 2))
    print('Total Grid unavoidable cost = $', round(np.sum(Grid_unavoidable_cost), 2))
    print('Total avoided costs = $', round(np.sum(avoided_costs_e), 2))
    print('Total net avoided costs by hybrid energy system = $', round(np.sum(avoided_costs_e) + np.sum(avoided_costs_ng) - np.sum(Grid_Cost_net) - np.sum(gHeating_Cost), 2))
    print('Total grid earning = $', round(np.sum(Sold_electricity), 2))
    print('Total grid costs = $', round(np.sum(Grid_Cost_net), 2))
    print('Total grid costs for the property = $', round(np.sum(Grid_Cost_p), 2))
    print('Total grid credits = $', round(np.sum(Total_grid_credits), 2))
    print('LCOE  =', round(LCOE, 2), '$/kWh')
    print('LCOE without incentives =', round(LCOE_without_incentives, 2), '$/kWh')
    print('LCOE for only Grid connected system =', round(LCOE_Grid, 2), '$/kWh')
    print('Grid avoidable cost per kWh =', round(Grid_avoidable_cost_perkWh, 2), '$/kWh')
    print('Grid unavoidable cost per kWh =', round(Grid_unavoidable_cost_perkWh, 2), '$/kWh')
    print('Solar Cost per kWh =', round(Solar_Cost_perkWh, 2), '$/kWh')
    print('Operating Cost  = $', round(Operating_Cost, 2))
    print('Initial Cost  = $', round(I_Cost, 2))
    print('Initial Cost without incentives= $', round(I_Cost_without_incentives, 2))
    print('Total incentives received= $', round(Total_incentives_received, 2))
    print('Total replacement cost  = $', round(np.sum(R_Cost_res), 2))
    print('Total operation and maintenance cost  = $', round(np.sum(MO_Cost), 2))
    print('Total Salvage cost  = $', round(np.sum(Salvage_S), 2))
    if Grid == 1:
        print('Total Money paid to the Grid= $', round(np.sum(Grid_Cost_net), 2))

    print('Total Money paid by the user= $', round(np.sum(NPC), 2))

    print(' ')
    text = "Technical Results"
    line = text.center(50, "=")
    print(line)
    print('PV Power  =', np.sum(Ppv), 'kWh')
    if np.sum(Pwt) > 0.1:
        print('WT Power  =', np.sum(Pwt), 'kWh')
    if np.sum(Pdg) > 0.1:
        print('Generator Power  =', np.sum(Pdg), 'kWh')
    if np.sum(Pch) or np.sum(Pdch) > 0.01:
        print('Battery Energy In  =', np.sum(Pch), 'kWh')
        print('Battery Energy Out  =', np.sum(Pdch), 'kWh')
    if EV > 0:
        print('EV Energy In  =', np.sum(Pev_ch), 'kWh')
        print('EV Energy Out  =', np.sum(Pev_dch), 'kWh')
    print('RE  =', round(100 * RE, 2), '%')
    print('LPSP Property  =', round(100 * LPSP_p, 2), '%')
    print('LPSP EV  =', round(100 * LPSP_ev, 2), '%')
    print('LPSP Total  =', round(100 * LPSP, 2), '%')
    print('Annual Property Load =', np.sum(Eload), 'kWh')
    print('Annual EV Load =', np.sum(EV_dem), 'kWh')
    print('Annual Total Load =', np.sum(Eload + EV_dem), 'kWh')
    print('Annual Total Served Load =', np.sum(Eload_served), 'kWh')
    print('Annual Property Capacity Shortage =', np.sum(Ens), 'kWh')
    print('Annual EV Capacity Shortage =', np.sum(Ens_EV), 'kWh')
    print('Excess Electricity =', np.sum(Edump), 'kWh')

    if Grid == 1:
        Total_Pbuy = (np.sum(Pbuy)) * (Grid > 0)
        Total_Psell = (np.sum(Psell)) * (Grid > 0)
        print('Annual power bought from Grid= ', Total_Pbuy, 'kWh')
        print('Annual Power sold to Grid= ', Total_Psell, 'kWh')
        print('Grid Emissions   =', Grid_Emissions, '(kg/year)')

    print('Grid Only Emissions   =', Grid_only_Emissions, '(kg/year)')
    if np.sum(Pdg) > 0.1:
        print('Generator Operating Hours   =', Top_DG, '(Hours/year)')
        print('Annual fuel consumed by Generator   =', np.sum(q), '(Liter/year)')
        print('Generator Emissions   =', DG_Emissions, '(kg/year)')
    print('LEM  =', LEM, 'kg/kWh')


    # Set the font to Times New Roman
    plt.rcParams["font.family"] = "Times New Roman"

    # Cash flow chart
    # Convert arrays to list
    R_Cost = [-x for x in R_Cost.tolist()]  # Flip the signs
    MO_Cost = [-x for x in MO_Cost.tolist()]  # Flip the signs
    C_Fu = [-x for x in C_Fu.tolist()]  # Flip the signs
    Salvage = Salvage.tolist()
    Grid_Cost = Grid_Cost_net.tolist()  # Convert to list
    Grid_Cost_pos = [-x if x > 0 else 0 for x in Grid_Cost]  # Only keep positive values and flip the sign
    Grid_Cost_neg = [-x if x < 0 else 0 for x in Grid_Cost]  # Only keep negative values and flip the sign
    avoided_costs_e = [x for x in avoided_costs_e.tolist()]
    avoided_costs_ng = [x for x in avoided_costs_ng.tolist()]

    # Define years from 0 to n
    years = list(range(n + 1))

    # Calculate the yearly total cost for year 1 to n (without including Salvage)
    yearly_total_cost = [sum(x) + g + a + b for x, g, a, b in zip(zip(R_Cost, MO_Cost, C_Fu, Salvage, Grid_Cost_pos), Grid_Cost_neg, avoided_costs_e, avoided_costs_ng)]
    yearly_total_cost = [-I_Cost] + yearly_total_cost  # Add initial investment (flipped) at year 0
    # Calculate the cumulative total cost
    cumulative_total_cost = [sum(yearly_total_cost[:i + 1]) for i in range(n + 1)]

    # Calculate IRR and Payback Period
    IRR_cost = yearly_total_cost
    cumulative_total_cost_PP = [sum(yearly_total_cost[:i + 1]) for i in range(n + 1)]

    irr = npf.irr(IRR_cost)
    if irr < 0:
        print("The projected investment is a loss")
        print(f"The IRR of the project is: {irr:.2%}")
    else:
        print(f"The IRR of the project is: {irr:.2%}")

    payback_period = next((i for i, cost in enumerate(cumulative_total_cost_PP) if cost >= 0), None)
    if payback_period is None:
        print("No payback period within the project's lifetime")
    else:
        print(f"The Payback Period is: {payback_period} years")

    # Calculate Total Revenues and Total Costs
    total_revenues = sum(avoided_costs_e) + sum(avoided_costs_ng) + (sum(Salvage) * (1 + System_Tax)) + sum(Grid_Cost_neg)
    print(f"Total revenues is: {total_revenues:.2f}")
    total_costs = (((I_Cost - sum(MO_Cost) - sum(R_Cost) - sum(C_Fu)) * (1 + System_Tax)) - sum(Grid_Cost_pos))

    # Calculate Net Profit
    net_profit = total_revenues - total_costs
    print(f"Total net profit is: {net_profit:.2f}")
    print(f"Total costs is: {total_costs:.2f}")
    # Calculate ROI
    roi = (net_profit / I_Cost) * 100

    print(f"The ROI of the project is: {roi:.2f}%")
    #irr2 = npf.irr([-15318.53,	2464.45179,	2433.920272,	2403.692704,	2373.76884,	2344.148256,	2314.83044,	2285.814768,	2257.10048,	2228.686728,	2200.572576,	2172.756976,	2145.238808,	2118.01684,	2091.089808,	2064.45632,	2038.114944,	2012.064176,	1986.302456,	1960.828136,	1935.639552,	1910.734944,	1886.112544,	1861.770504,	1837.706968,	1813.919992])
    #print(irr2)
    # Create the bar chart
    plt.figure(figsize=(10, 6))

    # Plot costs conditionally
    if any(R_Cost):
        plt.bar(years[1:], R_Cost, label='Replacement Cost', color='blue')
    if any(MO_Cost):
        plt.bar(years[1:], MO_Cost, bottom=R_Cost, label='Maintenance & Operating Cost', color='brown')
    if any(C_Fu):
        plt.bar(years[1:], C_Fu, bottom=[i + j for i, j in zip(R_Cost, MO_Cost)], label='Fuel Cost', color='orange')
    if any(Grid_Cost_pos):
        plt.bar(years[1:], Grid_Cost_pos, bottom=[i + j + k for i, j, k in zip(R_Cost, MO_Cost, C_Fu)], label='Grid Cost', color='purple')

    # Plot grid revenues
    if any(Grid_Cost_neg):
        plt.bar(years[1:], Grid_Cost_neg, label='Grid Revenue', color='pink', bottom=0)
    # Plot avoided costs as revenue above x-axis
    if any(avoided_costs_e):
        plt.bar(years[1:], avoided_costs_e, bottom=Grid_Cost_neg, label='Avoided Costs', color='cyan', alpha=1)
    if any(avoided_costs_ng):
        plt.bar(years[1:], avoided_costs_ng, bottom=[i + j for i, j in zip(Grid_Cost_neg, avoided_costs_e)], label='Avoided Costs NG', color='darkgreen', alpha=1)
    # Plot salvage revenues (Start from x-axis)
    if any(Salvage):
        plt.bar(years[-1:], Salvage, bottom=[i + j + k for i, j, k in zip(Grid_Cost_neg, avoided_costs_e, avoided_costs_ng)], label='Salvage', color='green')

    # Initial capital cost
    plt.bar(0, -I_Cost, label='Initial Investment', color='red')  # Flip the sign

    # Plot total cost curve
    plt.plot(years, cumulative_total_cost, color='black', marker='o', label='Total Cost')

    # Add details and labels
    #plt.title('Cash Flow Chart', fontsize=20)
    plt.xlabel('Year', fontsize=16)
    plt.ylabel('Cash Flow [$]', fontsize=16)
    plt.xticks([i for i in years], years, fontsize=14)
    plt.legend(fontsize=12)
    # Make x-axis visible
    plt.axhline(0, color='black', linewidth=0.8)
    # Customize y-axis ticks
    ax = plt.gca()
    ax.yaxis.set_major_locator(mticker.MaxNLocator(integer=True))  # Automatically determines nice y-tick locations
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f'{int(x):,}'))  # Formats the y-tick labels with commas
    plt.yticks(fontname='Times New Roman', fontsize=14)

    y_min, y_max = plt.ylim()
    y_margin = (y_max - y_min) * 0.025
    plt.ylim(y_min - y_margin, y_max + y_margin)
    #plt.title('c) Off Grid - Day Shift - Singapore', fontsize=16, fontweight='bold', loc='left')
    plt.tight_layout()
    plt.savefig('output/figs/Cash Flow.svg', dpi=300)

    # Cash flow chart ADVANCED

    # Define years from 0 to n
    years = list(range(n + 1))

    # Calculate the yearly total cost for year 1 to n (without including Salvage)
    yearly_total_only_grid_cost = -Grid_Cost_onlyG
    yearly_total_only_grid_cost = yearly_total_only_grid_cost.tolist()
    yearly_total_only_grid_cost = [0] + yearly_total_only_grid_cost  # Add initial investment (flipped) at year 0
    # Calculate the cumulative total cost
    cumulative_total_only_grid_cost = [sum(yearly_total_only_grid_cost[:i + 1]) for i in range(n + 1)]

    # Calculate the yearly total cost for year 1 to n (without including Salvage)
    yearly_total_grid_cost = [sum(x) + g for x, g in zip(zip(R_Cost, MO_Cost, C_Fu, Salvage, Grid_Cost_pos), Grid_Cost_neg)]
    # Calculate the cumulative total cost
    yearly_total_grid_cost = [-I_Cost] + yearly_total_grid_cost
    # Calculate the cumulative total cost
    cumulative_total_grid_cost = [sum(yearly_total_grid_cost[:i + 1]) for i in range(n + 1)]


    # Create the bar chart
    plt.figure(figsize=(10, 6))

    # Plot costs conditionally
    if any(R_Cost):
        plt.bar(years[1:], R_Cost, label='Replacement Cost', color='blue')
    if any(MO_Cost):
        plt.bar(years[1:], MO_Cost, bottom=R_Cost, label='Maintenance & Operating Cost', color='brown')
    if any(C_Fu):
        plt.bar(years[1:], C_Fu, bottom=[i + j for i, j in zip(R_Cost, MO_Cost)], label='Fuel Cost', color='orange')
    if any(Grid_Cost_pos):
        plt.bar(years[1:], Grid_Cost_pos, bottom=[i + j + k for i, j, k in zip(R_Cost, MO_Cost, C_Fu)], label='Grid Cost', color='purple')

    # Plot grid revenues
    if any(Grid_Cost_neg):
        plt.bar(years[1:], Grid_Cost_neg, label='Grid Revenue', color='pink', bottom=0)
    # Plot avoided costs as revenue above x-axis
    if any(avoided_costs_e):
        plt.bar(years[1:], avoided_costs_e, bottom=Grid_Cost_neg, label='Avoided Costs', color='cyan', alpha=1)
    if any(avoided_costs_ng):
        plt.bar(years[1:], avoided_costs_ng, bottom=[i + j for i, j in zip(Grid_Cost_neg, avoided_costs_e)], label='Avoided Costs_NG', color='darkgreen', alpha=1)
    # Plot salvage revenues (Start from x-axis)
    if any(Salvage):
        plt.bar(years[-1:], Salvage, bottom=[i + j + k for i, j, k in zip(Grid_Cost_neg, avoided_costs_e, avoided_costs_ng)], label='Salvage', color='green')

    # Initial capital cost
    plt.bar(0, -I_Cost, label='Initial Investment', color='gray')  # Flip the sign

    # Plot total cost curve (without grid)
    plt.plot(years, cumulative_total_only_grid_cost, color='black', marker='o', label='Total Only Grid Cost')
    # Plot total cost curve (with grid)
    plt.plot(years, cumulative_total_grid_cost, color='red', marker='s', label='Total Customer Cost')

    # Add details and labels
    # plt.title('Cash Flow Chart', fontsize=20)
    plt.xlabel('Year', fontsize=16)
    plt.ylabel('Cash Flow [$]', fontsize=16)
    plt.xticks([i for i in years], years, fontsize=14)
    plt.legend(fontsize=12)
    # Make x-axis visible
    plt.axhline(0, color='black', linewidth=0.8)
    # Customize y-axis ticks
    ax = plt.gca()
    ax.yaxis.set_major_locator(mticker.MaxNLocator(integer=True))  # Automatically determines nice y-tick locations
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f'{int(x):,}'))  # Formats the y-tick labels with commas
    plt.yticks(fontname='Times New Roman', fontsize=14)

    y_min, y_max = plt.ylim()
    y_margin = (y_max - y_min) * 0.025
    plt.ylim(y_min - y_margin, y_max + y_margin)
    plt.tight_layout()
    plt.savefig('output/figs/Cash Flow_ADV.png', dpi=300)

    # Advanced multi-Cash Flow Chart
    ########################
    if Cash_Flow_adv == 1:
        from matplotlib import pyplot
        # Function to save data to CSV file
        def save_data_to_csv(filename, data):
            with open(filename, 'a', newline='') as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow(data)

        # Function to load data from CSV file
        def load_data_from_csv(filename):
            data = []
            with open(filename, 'r') as csvfile:
                reader = csv.reader(csvfile)
                for row in reader:
                    data.append(list(map(eval, row)))
            return data

        csv_filename = 'cash_flow_data.csv'

        # Save data to CSV
        data = [R_Cost, MO_Cost, C_Fu, Salvage, avoided_costs_e, avoided_costs_ng, Grid_Cost_pos, Grid_Cost_neg, I_Cost]
        save_data_to_csv(csv_filename, data)

        # Load all data from CSV
        all_data = load_data_from_csv(csv_filename)

        # Define colors and hatch patterns
        colors_curve = ['black', 'blue', 'purple', 'red', 'yellow', 'green']

        plt.figure(figsize=(10, 6))
        bar_legend_handles = []  # List to store handles for bar legends
        curve_labels = []  # List to store labels for curve legends
        curve_handles = []  # List to store handles for curve legends

        for idx, data in enumerate(all_data):
            R_Cost, MO_Cost, C_Fu, Salvage, avoided_costs_e, avoided_costs_ng, Grid_Cost_pos, Grid_Cost_neg, I_Cost = data
            years = list(range(n + 1))
            yearly_total_cost = [sum(x) + g + a + b for x, g, a, b in zip(zip(R_Cost, MO_Cost, C_Fu, Salvage, Grid_Cost_pos), Grid_Cost_neg, avoided_costs_e, avoided_costs_ng)]
            yearly_total_cost = [-I_Cost] + yearly_total_cost
            cumulative_total_cost = [sum(yearly_total_cost[:i + 1]) for i in range(n + 1)]

            # Adjust x-axis positions for bars
            bar_width = 0.075  # to adjust the width of bars
            offset = bar_width * idx  # to shift bars horizontally
            bar_positions = [x + offset for x in years[1:]]

            # Plot costs
            if any(R_Cost):
                plt.bar(bar_positions, R_Cost, label='Replacement Cost', color='blue', edgecolor='black', width=0.5)
            if any(MO_Cost):
                plt.bar(bar_positions, MO_Cost, bottom=R_Cost, label='Maintenance & Operating Cost', color='brown', edgecolor='black', width=0.5)
            if any(C_Fu):
                plt.bar(bar_positions, C_Fu, bottom=[i + j for i, j in zip(R_Cost, MO_Cost)], label='Fuel Cost', color='orange', edgecolor='black', width=0.5)
            if any(Grid_Cost_pos):
                plt.bar(bar_positions, Grid_Cost_pos, bottom=[i + j + k for i, j, k in zip(R_Cost, MO_Cost, C_Fu)], label='Grid Cost', color='purple', edgecolor='black', width=0.5)

            # Plot grid revenues
            if any(Grid_Cost_neg):
                plt.bar(bar_positions, Grid_Cost_neg, label='Grid Revenue', color='pink', alpha=1, edgecolor='black', width=0.5)
            if any(avoided_costs_e):
                plt.bar(bar_positions, avoided_costs_e, bottom=Grid_Cost_neg, label='Avoided Costs', color='cyan', alpha=1, edgecolor='black', width=0.5)
            if any(avoided_costs_ng):
                plt.bar(bar_positions, avoided_costs_ng, bottom=[i + j for i, j in zip(Grid_Cost_neg, avoided_costs_e)], label='Avoided Costs', color='darkgreen', alpha=1, edgecolor='black', width=0.5)
            # Plot salvage revenues (Start from x-axis)
            if any(Salvage):
                plt.bar(bar_positions, Salvage, bottom=[i + j + k for i, j, k in zip(Grid_Cost_neg, avoided_costs_e, avoided_costs_ng)], label='Salvage', color='green', edgecolor='black', width=0.5)

            # Plot initial investment as a red bar
            plt.bar(0, -I_Cost, label='Initial Investment' if idx == 0 else None, color='red', edgecolor='black', width = 0.5)

            # Store handles for legends of the bars in the first loop iteration
            if idx == 0:
                bar_legend_handles, _ = plt.gca().get_legend_handles_labels()
            LG_lines = [7.5, 10, 12.5, 15, 17.5, 20]
            # Plot curves with different colors and names
            color_idx = idx % len(colors_curve)  # Ensure idx stays within the bounds of colors_curve
            curve_labels.append(f'Total System Cost (GER =  {idx*2}%)')
            curve_handles.extend(plt.plot(years, cumulative_total_cost, linestyle='-', marker='o', linewidth=0.8, color=colors_curve[color_idx]))

        # Plot legends for bars outside of the loop
        legend1 = plt.legend(handles=bar_legend_handles)
        # Plot legends for curves
        plt.legend(curve_handles, curve_labels, loc='center left')
        pyplot.gca().add_artist(legend1)
        # Add details and labels
        plt.xlabel('Year', fontsize=16)
        plt.ylabel('Cash Flow [$]', fontsize=16)
        plt.xticks([i for i in years], years, fontsize=14)
        plt.axhline(0, color='black', linewidth=0.8)
        # Customize y-axis ticks
        ax = plt.gca()
        ax.yaxis.set_major_locator(mticker.MaxNLocator(integer=True))  # Automatically determines nice y-tick locations
        ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f'{int(x):,}'))  # Formats the y-tick labels with commas
        plt.yticks(fontname='Times New Roman', fontsize=14)
        # Set y-axis limit
        y_min, y_max = plt.ylim()
        y_margin = (y_max - y_min) * 0.025
        plt.ylim(y_min - y_margin, y_max + y_margin)
        plt.tight_layout()
        plt.savefig('output/figs/Multiple_Cash_Flow_ADV.png', dpi=300)

    from Electricity_Bill_Calculator import compare_scenarios, print_comparison

    Pbuy_no_sell = Eload + EV_dem  # No generation, buy everything
    Pbuy_with_sell = Pbuy  # With generation, buy only net demand

    # Run comparison
    comparison = compare_scenarios(daysInMonth, Pbuy_no_sell,  Pbuy_with_sell,  Cbuy, Service_charge, Grid_Tax, Grid_Tax_amount, Annual_expenses, Grid_escalation, Grid_credit, NEM_fee, Grid, n, ir, Psell, Csell)
    # Print comparison
    print_comparison(comparison, show_plot=True)

    #Grid purchase and sale figure
    if Grid == 0:
        Pbuy = 0
        Psell = 0
    else:
        fig, ax = plt.subplots(figsize=(14, 10))  # Increased figure size and resolution
        ax.plot(Pbuy, marker='.', linestyle='-', linewidth=0.5, color='blue',label='Buy')  # Added markers and reduced line width
        ax.plot(Psell, marker='.', linestyle='-', linewidth=0.5, color='red',label='Sell')  # Added markers and reduced line width
        ax.set_ylabel('Pgrid [kW]', fontsize=22)  # Increased font size
        ax.set_xlabel('Month', fontsize=22)  # Increased font size and changed label to 'Month'
        ax.tick_params(axis='both', which='major', labelsize=18)  # Increased tick label size
        ax.legend(loc='upper left', bbox_to_anchor=(1, 1), fontsize=22)  # Moved legend closer to the figure
        ax.set_title('Power Purchase vs Sale', fontsize=22)  # Increased font size
        # Assuming a non-leap year (365 days), set the x-ticks at the beginning of each month.
        hours_per_month = [0, 31 * 24, 59 * 24, 90 * 24, 120 * 24, 151 * 24, 181 * 24, 212 * 24, 243 * 24, 273 * 24,304 * 24, 334 * 24]
        month_labels = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
        ax.set_xticks(hours_per_month)
        ax.set_xticklabels(month_labels)
        # Set x-axis and y-axis to start from zero
        ax.set_xlim([0, max(hours_per_month) + 744])
        ax.set_ylim([0, max(max(Pbuy), max(Psell))])
        # Adjust the margins and space between subplots
        plt.subplots_adjust(left=0.05, right=0.85, top=0.95, bottom=0.08)
        plt.tight_layout()
        plt.savefig('output/figs/Grid Interconnection.png', dpi=300)

    # Energy/Power Distribution figure
    fig, ax = plt.subplots(figsize=(30, 10))  # Increased figure size and resolution
    # Increased linewidth and added distinct colors, line styles, and markers for visibility
    ax.plot(Eload_served, linestyle='-', linewidth=1, color='blue', marker='o', markersize=4, label='Served load')
    ax.plot(Pdg, linestyle='--', linewidth=1, color='red', marker='x', markersize=4, label='$P_{dg}$')
    ax.plot(P_bat, linestyle='-.', linewidth=1, color='green', marker='^', markersize=4, label='$P_{bat}$')
    ax.plot(P_RE, linestyle=':', linewidth=1, color='purple', marker='s', markersize=4, label='$P_{RE}$')
    ax.set_ylabel('Power [kW]', fontsize=22)  # Increased font size
    ax.set_xlabel('Month', fontsize=22)  # Increased font size
    ax.tick_params(axis='both', which='major', labelsize=18)  # Increased tick label size
    ax.legend(loc='upper left', bbox_to_anchor=(1, 1), fontsize=22)  # Moved legend closer to the figure
    #ax.set_title('Energy Distribution', fontsize=22)  # Increased font size
    # Set x-ticks
    hours_per_month = [0, 31 * 24, 59 * 24, 90 * 24, 120 * 24, 151 * 24, 181 * 24, 212 * 24, 243 * 24, 273 * 24, 304 * 24, 334 * 24]
    month_labels = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
    ax.set_xticks(hours_per_month)
    ax.set_xticklabels(month_labels)
    # Set x-axis to start from zero and y-axis according to data
    ax.set_xlim([0, max(hours_per_month) + 744])
    ax.set_ylim([0, max(max(Eload - Ens), max(Pdg), max(Pch - Pdch), max(Ppv + Pwt))])
    # Adjust the margins and space between subplots
    plt.subplots_adjust(left=0.08, right=0.78, top=0.95, bottom=0.08)
    plt.tight_layout()
    plt.savefig('output/figs/Energy Distribution.png', dpi=300)

    # State of charge figure
    if Nbat > 0:
        fig, ax = plt.subplots(figsize=(20, 10))  # Increased figure size and resolution
        # Plot 'State of Charge'
        ax.plot(Eb / Cn_B, marker='.', linestyle='-', linewidth=0.5, color='blue')
        ax.set_ylabel('SOC (%)', fontsize=22)  # Increased font size
        ax.set_xlabel('Month', fontsize=22)  # Increased font size
        ax.tick_params(axis='both', which='major', labelsize=18)  # Increased tick label size
        #ax.set_title('State of Charge', fontsize=22)  # Increased font size
        # Assuming a non-leap year (365 days), set the x-ticks at the beginning of each month.
        hours_per_month = [0, 31 * 24, 59 * 24, 90 * 24, 120 * 24, 151 * 24, 181 * 24, 212 * 24, 243 * 24, 273 * 24, 304 * 24, 334 * 24]
        month_labels = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
        ax.set_xticks(hours_per_month)
        ax.set_xticklabels(month_labels)
        # Set x-axis and y-axis to start from zero
        ax.set_xlim([0, max(hours_per_month) + 744])
        ax.set_ylim([0, max(Eb / Cn_B)])
        # Adjust the margins and space between subplots
        plt.subplots_adjust(left=0.08, right=0.95, top=0.95, bottom=0.08)
        plt.tight_layout()
        plt.savefig('output/figs/Battery State of Charge.png', dpi=300)

    # Plot results for one specific day
    # Function to filter out data series with sum less than 0.1 in the specified range
    def non_zero_data_series(data_series, t1, t2):
        return [(title, data, label) for title, data, label in data_series if np.sum(data[t1:t2 + 1]) >= 0.1]

    # Specific day number
    Day = 180
    t1 = Day * 24
    t2 = (Day + 1) * 24

    data_series = [('Load Profile', Eload, '$E_{load}$ [kW]'),
                   ('Plane of Array Irradiance', G, '$POA [W/m^{2}]$'),
                   ('Ambient Temperature', T, '$T [^{o}C]$'),
                   ('PV Power', Ppv, '$P_{pv}$ [kW]'),
                   ('WT Power', Pwt, '$P_{wt}$ [kW]'),
                   ('Diesel Generator Power', Pdg, '$P_{DG}$ [kW]'),
                   ('Battery Energy Level', Eb, '$E_{b}$ [kWh]'),
                   ('State of Charge', Eb / Cn_B if not np.all(Eb[t1:t2] == 0) else np.zeros_like(Eb), 'SOC (%)'),
                   ('Loss of Power Supply', Ens, 'LPS[kWh]'),
                   ('Dumped Energy', Edump, '$E_{dump}$ [kWh]'),
                   ('Battery discharge Power', Pdch, '$P_{dch}$ [kW]'),
                   ('Battery charge Power', Pch, '$P_{ch}$ [kW]')]

    # Apply the filter function to remove series with sum less than 0.1
    non_zero_series = non_zero_data_series(data_series, t1, t2)

    n_non_zeros = len(non_zero_series)  # Number of non-zero series
    n_cols = 3  # Number of columns
    n_rows = n_non_zeros // n_cols + (n_non_zeros % n_cols > 0)  # Number of rows

    fig, axs = plt.subplots(n_rows, n_cols, figsize=(16, n_rows * 4))  # Adjusting figure size based on number of rows
    fig.suptitle('Results for ' + str(Day) + '-th day', fontsize=24)  # Increase font size
    axs = axs.flatten()  # Flatten the 2D array to 1D for easy iteration

    for i, (title, data, label) in enumerate(non_zero_series):
        ax = axs[i]
        ax.plot(range(t1, t2), data[t1:t2], linewidth=2)  # Increase line width
        ax.set_title(title, fontsize=16)  # Increase font size
        ax.set_ylabel(label, fontsize=16)  # Increase font size
        ax.set_xlabel('t[hour]', fontsize=16)  # Increase font size
        ax.tick_params(axis='both', which='major', labelsize=12)  # Increase font size

    for j in range(i + 1, n_rows * n_cols):  # Turn off unused subplots
        axs[j].axis('off')

    plt.tight_layout(rect=[0, 0, 1, 0.96])  # Adjust the layout to make space for the title
    plt.savefig('output/figs/Specific day results.png', dpi=300)

    # Utility figures

    A_l = np.zeros((12, 31))
    index = 1
    for m in range(12):
        index1 = index
        for d in range(daysInMonth[m]):
            Total_daily_load = np.sum(Eload[index1:index1 + 23])
            A_l[m, d] = Total_daily_load
            index1 = index1 + 24
        index = (24 * daysInMonth[m]) + index

    # The figure showing each day/month/year average cost of energy system
    A_l_served = np.zeros((12, 31))
    index = 1
    for m in range(12):
        index1 = index
        for d in range(daysInMonth[m]):
            Total_daily_served_load = np.sum(Eload_served[index1:index1 + 23])
            A_l_served[m, d] = Total_daily_served_load
            index1 = index1 + 24
        index = (24 * daysInMonth[m]) + index

    # Justified LCOE for average cost of system
    LCOE_justified = CRF * NPC / np.sum(Eload_served)  # Levelized Cost of Energy ($/kWh)

    if np.sum(Psell) > 0.1:
        AE_c = np.round(LCOE_justified * A_l_served, 2)
    else:
        AE_c = np.round(LCOE * A_l_served, 2)
    # Compute monthly sums
    sums = np.sum(AE_c, axis=1, keepdims=True)
    yearly_sum = np.nansum(AE_c)
    AE_c = np.hstack((AE_c, sums))

    # Plotting heat map of each day average cost of energy system in each month
    AE_c[np.where(AE_c == 0)] = np.nan
    # Increase the figure size
    fig = plt.figure(figsize=(20, 15))
    # Define grid and increase the space between heatmap and colorbar
    gs = gridspec.GridSpec(2, 2, width_ratios=[19, 1], height_ratios=[19, 1])
    gs.update(wspace=0.05, hspace=0.1)  # Increase hspace
    ax0 = plt.subplot(gs[0])
    # Increase the size of the annotation and set linewidths to 0
    sns.heatmap(AE_c[:, :-1], cmap='jet', annot=True, fmt=".1f", yticklabels=False, ax=ax0, cbar=False, annot_kws={"size": 18}, linewidths=0)
    # Set the y-tick labels
    months = ["January", "February", "March", "April", "May", "June", "July", "August", "September", "October", "November", "December"]
    ax0.set_yticks(np.arange(len(months)) + 0.5)  # Centering the labels
    ax0.set_yticklabels(months)
    # Adjust the x-ticks and labels to start from 1
    xticks = np.arange(1, AE_c.shape[1])
    ax0.set_xticks(xticks - 0.5, minor=False)  # Shift the ticks to be centered
    ax0.set_xticklabels(list(range(1, AE_c.shape[1])))
    # Increase the size of y and x tick labels
    ax0.tick_params(axis='y', labelsize=22)
    ax0.tick_params(axis='x', labelsize=22)

    # Plotting the Monthly column
    ax1 = plt.subplot(gs[1])
    sns.heatmap(AE_c[:, -1:], cmap='jet', annot=True, fmt=".1f", yticklabels=False, ax=ax1, cbar=False, annot_kws={"size": 18}, linewidths=0)
    ax1.set_xticks([0.5])
    ax1.set_xticklabels(['Monthly'])
    ax1.tick_params(axis='x', labelsize=22)
    # Synchronize y-axis limits
    ax1.set_ylim(ax0.get_ylim())
    ax2 = plt.subplot(gs[2])
    cb = plt.colorbar(ax0.collections[0], cax=ax2, orientation='horizontal')
    # Increase the size of color bar tick labels
    ax2.tick_params(labelsize=22)
    # Place the title below the color bar
    ax2.set_title('Daily average cost of energy system [$]', fontsize=32, y=-1.35)
    # Add yearly sum as a new cell at the bottom right of the figure
    pos = ax2.get_position()
    pos = [pos.x1 + 0.035, pos.y0 - 0.035, 0.08, pos.height]
    ax3 = fig.add_axes(pos)
    text = ax3.text(0.5, 0.5, f'Yearly: {yearly_sum:.1f}', horizontalalignment='center', verticalalignment='center', fontsize=32)
    ax3.axis('off')
    # Add a separate color bar for the 'Monthly' column
    # Get position of ax1 subplot in figure coordinate
    pos = ax1.get_position()
    cax = fig.add_axes([pos.x1 + 0.01, pos.y0 - 0.02, 0.02, pos.height + 0.12])  # position for the colorbar [left, bottom, width, height]
    cbar_total = plt.colorbar(ax1.collections[0], cax=cax, orientation='vertical')
    cbar_total.ax.tick_params(labelsize=22)
    cbar_total.ax.set_title('Monthly average cost of energy system [$]', fontsize=32, rotation=270, x=3.5, y=0.16)
    fig.subplots_adjust(left=0.075, top=0.98, bottom=0.075)
    plt.savefig('output/figs/Daily-Monthly-Yearly average cost of energy system.png', dpi=300)

    # Calculate average hourly grid cost for each day in each month
    Gh_c = np.zeros((12, 31))
    index = 1
    for m in range(12):
        index1 = index
        for d in range(daysInMonth[m]):
            gridcost = ((np.mean(Cbuy[index1:index1 + 23])) * (1 + Grid_Tax)) + Grid_Tax_amount
            Gh_c[m, d] = gridcost
            index1 = index1 + 24
        index = (24 * daysInMonth[m]) + index

    # Compute monthly sums
    averages_grid_hourly_cost = np.nanmean(Gh_c, axis=1, keepdims=True)
    yearly_mean = np.nanmean(Gh_c)
    Gh_c = np.hstack((Gh_c, averages_grid_hourly_cost))

    # Plot average hourly Grid cost (Cbuy) for each day in each month heatmap
    Gh_c[np.where(Gh_c == 0)] = np.nan
    # Increase the figure size
    fig = plt.figure(figsize=(20, 15))
    # Define grid and increase the space between heatmap and colorbar
    gs = gridspec.GridSpec(2, 2, width_ratios=[19, 1], height_ratios=[19, 1])
    gs.update(wspace=0.05, hspace=0.1)  # Increase hspace
    ax0 = plt.subplot(gs[0])
    # Increase the size of the annotation and set linewidths to 0
    sns.heatmap(Gh_c[:, :-1], cmap='jet', annot=True, fmt=".1f", yticklabels=False, ax=ax0, cbar=False, annot_kws={"size": 18}, linewidths=0)
    # Set the y-tick labels
    months = ["January", "February", "March", "April", "May", "June", "July", "August", "September", "October", "November", "December"]
    ax0.set_yticks(np.arange(len(months)) + 0.5)  # Centering the labels
    ax0.set_yticklabels(months)
    # Adjust the x-ticks and labels to start from 1
    xticks = np.arange(1, Gh_c.shape[1])
    ax0.set_xticks(xticks - 0.5, minor=False)  # Shift the ticks to be centered
    ax0.set_xticklabels(list(range(1, Gh_c.shape[1])))
    # Increase the size of y and x tick labels
    ax0.tick_params(axis='y', labelsize=22)
    ax0.tick_params(axis='x', labelsize=22)

    # Plotting the Monthly column
    ax1 = plt.subplot(gs[1])
    sns.heatmap(Gh_c[:, -1:], cmap='jet', annot=True, fmt=".1f", yticklabels=False, ax=ax1, cbar=False, annot_kws={"size": 18}, linewidths=0)
    ax1.set_xticks([0.5])
    ax1.set_xticklabels(['Monthly'])
    ax1.tick_params(axis='x', labelsize=22)
    # Synchronize y-axis limits
    ax1.set_ylim(ax0.get_ylim())
    ax2 = plt.subplot(gs[2])
    cb = plt.colorbar(ax0.collections[0], cax=ax2, orientation='horizontal')
    # Increase the size of color bar tick labels
    ax2.tick_params(labelsize=22)
    # Place the title below the color bar
    ax2.set_title('Daily average hourly cost of connecting to the grid [$/kWh]', fontsize=32, y=-1.35)
    # Add yearly sum as a new cell at the bottom right of the figure
    pos = ax2.get_position()
    pos = [pos.x1 + 0.035, pos.y0 - 0.035, 0.08, pos.height]
    ax3 = fig.add_axes(pos)
    text = ax3.text(0.5, 0.5, f'Yearly: {yearly_mean:.1f}', horizontalalignment='center', verticalalignment='center', fontsize=32)
    ax3.axis('off')
    # Add a separate color bar for the 'Monthly' column
    # Get position of ax1 subplot in figure coordinate
    pos = ax1.get_position()
    cax = fig.add_axes([pos.x1 + 0.01, pos.y0 - 0.02, 0.02, pos.height + 0.12])  # position for the colorbar [left, bottom, width, height]
    cbar_total = plt.colorbar(ax1.collections[0], cax=cax, orientation='vertical')
    cbar_total.ax.tick_params(labelsize=22)
    cbar_total.ax.set_title('Monthly average hourly cost of connecting to the grid [$/kWh]', fontsize=32, rotation=270, x=3.85, y=0.04)
    fig.subplots_adjust(left=0.075, right=0.9, top=0.98, bottom=0.075)
    plt.savefig('output/figs/Daily-Monthly-Yearly average hourly cost of connecting to the grid.png', dpi=300)

    # Calculate average only grid connected system cost for each day/month/year
    AG_c = np.round(LCOE_Grid * A_l, 2)

    # Compute monthly sums
    sums = np.sum(AG_c, axis=1, keepdims=True)
    yearly_sum = np.nansum(AG_c)
    AG_c = np.hstack((AG_c, sums))

    # Plot average only grid connected system cost heatmap for each day in each month
    AG_c[np.where(AG_c == 0)] = np.nan
    # Increase the figure size
    fig = plt.figure(figsize=(20, 15))
    # Define grid and increase the space between heatmap and colorbar
    gs = gridspec.GridSpec(2, 2, width_ratios=[19, 1], height_ratios=[19, 1])
    gs.update(wspace=0.05, hspace=0.1)  # Increase hspace
    ax0 = plt.subplot(gs[0])
    # Increase the size of the annotation and set linewidths to 0
    sns.heatmap(AG_c[:, :-1], cmap='jet', annot=True, fmt=".1f", yticklabels=False, ax=ax0, cbar=False, annot_kws={"size": 18}, linewidths=0)
    # Set the y-tick labels
    months = ["January", "February", "March", "April", "May", "June", "July", "August", "September", "October", "November", "December"]
    ax0.set_yticks(np.arange(len(months)) + 0.5)  # Centering the labels
    ax0.set_yticklabels(months)
    # Adjust the x-ticks and labels to start from 1
    xticks = np.arange(1, AG_c.shape[1])
    ax0.set_xticks(xticks - 0.5, minor=False)  # Shift the ticks to be centered
    ax0.set_xticklabels(list(range(1, AG_c.shape[1])))
    # Increase the size of y and x tick labels
    ax0.tick_params(axis='y', labelsize=22)
    ax0.tick_params(axis='x', labelsize=22)

    # Plotting the Monthly column
    ax1 = plt.subplot(gs[1])
    sns.heatmap(AG_c[:, -1:], cmap='jet', annot=True, fmt=".1f", yticklabels=False, ax=ax1, cbar=False, annot_kws={"size": 18}, linewidths=0)
    ax1.set_xticks([0.5])
    ax1.set_xticklabels(['Monthly'])
    ax1.tick_params(axis='x', labelsize=22)
    # Synchronize y-axis limits
    ax1.set_ylim(ax0.get_ylim())
    ax2 = plt.subplot(gs[2])
    cb = plt.colorbar(ax0.collections[0], cax=ax2, orientation='horizontal')
    # Increase the size of color bar tick labels
    ax2.tick_params(labelsize=22)
    # Place the title below the color bar
    ax2.set_title('Daily average cost of only Grid-connected system [$]', fontsize=32, y=-1.35)
    # Add yearly sum as a new cell at the bottom right of the figure
    pos = ax2.get_position()
    pos = [pos.x1 + 0.035, pos.y0 - 0.035, 0.08, pos.height]
    ax3 = fig.add_axes(pos)
    text = ax3.text(0.5, 0.5, f'Yearly: {yearly_sum:.1f}', horizontalalignment='center', verticalalignment='center', fontsize=32)
    ax3.axis('off')
    # Add a separate color bar for the 'Monthly' column
    # Get position of ax1 subplot in figure coordinate
    pos = ax1.get_position()
    cax = fig.add_axes([pos.x1 + 0.01, pos.y0 - 0.02, 0.02, pos.height + 0.12])  # position for the colorbar [left, bottom, width, height]
    cbar_total = plt.colorbar(ax1.collections[0], cax=cax, orientation='vertical')
    cbar_total.ax.tick_params(labelsize=22)
    cbar_total.ax.set_title('Monthly average cost of only grid-connected system [$]', fontsize=32, rotation=270, x=3.5, y=0.09)
    fig.subplots_adjust(left=0.075, top=0.98, bottom=0.075)
    plt.savefig('output/figs/Daily-Monthly-Yearly average cost of only grid-connected system.png', dpi=300)

    # Hourly Grid electricity price color bar map
    # Assuming Cbuy is a 1D numpy array
    Cbuy_2D = np.reshape(Cbuy * (1 + Grid_Tax) + Grid_Tax_amount, (1, len(Cbuy)))  # Reshape to 2D
    fig, ax = plt.subplots(figsize=(10, 2), dpi=300)  # Increase figure size and resolution
    img = ax.imshow(Cbuy_2D, cmap='jet', aspect='auto')  # Display the data
    cbar = fig.colorbar(img, ax=ax, orientation='horizontal', pad=0.4, shrink=0.8)  # Add a colorbar and adjust its position
    cbar.set_label('Cbuy', size=15, labelpad=10)  # Add a label to colorbar and adjust its size and position
    # Set y ticks and labels empty
    ax.set_yticks([])
    # Increase x-tick label size
    ax.tick_params(axis='x', labelsize=15)
    # Add a title
    ax.set_title('Grid Hourly Cost [$/kWh]', fontsize=20)
    fig.subplots_adjust(left=0.05, right=0.95, top=0.85)  # Adjust the left, right, and top space
    # Assuming a non-leap year (365 days), set the x-ticks at the beginning of each month.
    hours_per_month = [0, 31 * 24, 59 * 24, 90 * 24, 120 * 24, 151 * 24, 181 * 24, 212 * 24, 243 * 24, 273 * 24, 304 * 24, 334 * 24]
    month_labels = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
    ax.set_xticks(hours_per_month)
    ax.set_xticklabels(month_labels)
    plt.savefig('output/figs/Grid Hourly Cost.png', dpi=300)

    # Calculate average money earned by selling electricity to grid in each day/month/year
    # Calculate average hourly grid sell for each day in each month
    Gh_s = np.zeros((12, 31))
    index = 1
    for m in range(12):
        index1 = index
        for d in range(daysInMonth[m]):
            gridsell = np.mean(Csell[index1:index1 + 23])
            Gh_s[m, d] = gridsell
            index1 = index1 + 24
        index = (24 * daysInMonth[m]) + index

    if np.sum(Psell) > 0.1:
        AG_s = np.zeros((12, 31))
        index = 1
        for m in range(12):
            index1 = index
            for d in range(daysInMonth[m]):
                Total_daily_sell = np.sum(Psell[index1:index1 + 23])
                AG_s[m, d] = Total_daily_sell
                index1 = index1 + 24
            index = (24 * daysInMonth[m]) + index

        AG_sc = np.round(Gh_s * AG_s, 2)

        # Compute monthly sums
        sums = np.sum(AG_sc, axis=1, keepdims=True)
        yearly_sum = np.nansum(AG_sc)
        AG_sc = np.hstack((AG_sc, sums))

        # Plot average money earned by selling electricity to grid in each day in each month heatmap
        AG_sc[np.where(AG_sc == 0)] = np.nan
        # Increase the figure size
        fig = plt.figure(figsize=(20, 15))
        # Define grid and increase the space between heatmap and colorbar
        gs = gridspec.GridSpec(2, 2, width_ratios=[19, 1], height_ratios=[19, 1])
        gs.update(wspace=0.05, hspace=0.1)  # Increase hspace
        ax0 = plt.subplot(gs[0])
        # Increase the size of the annotation and set linewidths to 0
        sns.heatmap(AG_sc[:, :-1], cmap='jet', annot=True, fmt=".1f", yticklabels=False, ax=ax0, cbar=False, annot_kws={"size": 18}, linewidths=0)
        # Set the y-tick labels
        months = ["January", "February", "March", "April", "May", "June", "July", "August", "September", "October", "November", "December"]
        ax0.set_yticks(np.arange(len(months)) + 0.5)  # Centering the labels
        ax0.set_yticklabels(months)
        # Adjust the x-ticks and labels to start from 1
        xticks = np.arange(1, AG_sc.shape[1])
        ax0.set_xticks(xticks - 0.5, minor=False)  # Shift the ticks to be centered
        ax0.set_xticklabels(list(range(1, AG_sc.shape[1])))
        # Increase the size of y and x tick labels
        ax0.tick_params(axis='y', labelsize=22)
        ax0.tick_params(axis='x', labelsize=22)

        # Plotting the Monthly column
        ax1 = plt.subplot(gs[1])
        sns.heatmap(AG_sc[:, -1:], cmap='jet', annot=True, fmt=".1f", yticklabels=False, ax=ax1, cbar=False, annot_kws={"size": 18}, linewidths=0)
        ax1.set_xticks([0.5])
        ax1.set_xticklabels(['Monthly'])
        ax1.tick_params(axis='x', labelsize=22)
        # Synchronize y-axis limits
        ax1.set_ylim(ax0.get_ylim())
        ax2 = plt.subplot(gs[2])
        cb = plt.colorbar(ax0.collections[0], cax=ax2, orientation='horizontal')
        # Increase the size of color bar tick labels
        ax2.tick_params(labelsize=22)
        # Place the title below the color bar
        ax2.set_title('Daily average Sell earning to the Grid [$]', fontsize=32, y=-1.35)
        # Add yearly sum as a new cell at the bottom right of the figure
        pos = ax2.get_position()
        pos = [pos.x1 + 0.035, pos.y0 - 0.035, 0.08, pos.height]
        ax3 = fig.add_axes(pos)
        text = ax3.text(0.5, 0.5, f'Yearly: {yearly_sum:.1f}', horizontalalignment='center', verticalalignment='center', fontsize=32)
        ax3.axis('off')
        # Add a separate color bar for the 'Monthly' column
        # Get position of ax1 subplot in figure coordinate
        pos = ax1.get_position()
        cax = fig.add_axes([pos.x1 + 0.01, pos.y0 - 0.02, 0.02, pos.height + 0.12])  # position for the colorbar [left, bottom, width, height]
        cbar_total = plt.colorbar(ax1.collections[0], cax=cax, orientation='vertical')
        cbar_total.ax.tick_params(labelsize=22)
        cbar_total.ax.set_title('Monthly average Sell earning to the Grid [$]', fontsize=32, rotation=270, x=3.5, y=0.225)
        fig.subplots_adjust(left=0.075, top=0.98, bottom=0.075)
        plt.savefig('output/figs/Daily-Monthly-Yearly average earning Sell to the Grid.png', dpi=300)

    if EV > 0 :
        # EV figures
        plt.rcParams.update({'xtick.labelsize': 14, 'ytick.labelsize': 14})

        # Compute energy values
        Edc = Ppv + Pwt + Pdch + Pev_dch - Pch - Pev_ch
        E_unb = Eload - Pbuy + Psell - Pdg - (n_I * Edc) * (Edc > 0) - (Edc / n_I) * (Edc < 0)

        # Define month boundaries and labels
        month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                       'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
        month_start_hours = np.concatenate(([0], np.cumsum(daysInMonth)[:-1])) * 24


        # Create the figure and set the style
        plt.figure(figsize=(10, 6))
        plt.plot(-Edump, 'k-.', linewidth=2, label='Dumped Energy')
        plt.plot(Ens, 'r', linewidth=2, label='Load Power Shortage')
        plt.plot(Ens_EV, 'b', linewidth=2, label='EV Power Shortage')

        # Enhance the visualization
        plt.xlabel('Month', fontsize=14)
        plt.ylabel('Power [kW]', fontsize=14)
        plt.legend(fontsize=12)
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.xticks(month_start_hours, month_names, fontsize=12)
        plt.yticks(fontsize=12)

        plt.tight_layout()
        plt.savefig('output/figs/Energy Balance.png', dpi=300)

        Eev1 = Eev[0: 8760].copy()
        Eev1[EV_p == 1] = np.nan  # EV out of home

        # Create the figure
        fig, axes = plt.subplots(2, 1, figsize=(10, 8))

        # Subplot 1: Plot Eev and Eev1
        axes[0].plot(Eev[0: 8760], 'b-.', linewidth=1, label='EV in home')
        axes[0].plot(Eev1, 'r', linewidth=1, label='EV out of Home')
        axes[0].legend(fontsize=12)
        axes[0].set_title('EV Energy', fontsize=14)
        #axes[0].set_xlabel('Time Step', fontsize=12)
        axes[0].set_ylabel('Energy [kWh]', fontsize=12)
        axes[0].set_xticks(month_start_hours, month_names, fontsize=12)
        axes[0].grid(True, linestyle='--', alpha=0.6)

        # Subplot 2: Bar plot of Pev_ch and -Pev_dch
        axes[1].bar(np.arange(len(Pev_ch)), Pev_ch, label='Pch', color='blue', alpha=0.7)
        axes[1].bar(np.arange(len(Pev_ch)), -Pev_dch, label='Pdch', color='red', alpha=0.7)
        axes[1].legend(fontsize=12)
        axes[1].set_title('EV Charging and Discharging', fontsize=14)
        #axes[1].set_xlabel('Time Step', fontsize=12)
        axes[1].set_ylabel('Power [kW]', fontsize=12)
        axes[1].set_xticks(month_start_hours, month_names, fontsize=12)
        axes[1].grid(True, linestyle='--', alpha=0.6)

        # Adjust layout and show the plot
        plt.tight_layout()
        plt.savefig('output/figs/EV Energy.png', dpi=300)

        # ─────── Inputs ───────
        dd = 160  # target day index (1-based)
        tt = np.arange(24 * (dd - 1) + 1, 24 * (dd + 1) + 1)  # 48-hour window

        fig, axes = plt.subplots(2, 1, figsize=(10, 6), sharex=True)

        # ─────── FIRST SUBPLOT: SOC (%) ───────
        Eev4_modified = Eev[0:8760].copy()
        EV_p_modified = EV_p.copy()

        # Mark the first hour when EV returns as 'still out of home'
        for t in range(len(EV_p) - 1):
            if EV_p[t] == 0 and EV_p[t + 1] == 1:
                EV_p_modified[t + 1] = 0

        # Set NaN for out-of-home values
        Eev4_modified[EV_p_modified == 1] = np.nan

        # Plot SOC in blue and red
        axes[0].plot(tt, (Eev[tt] / C_ev) * 100, 'b-.', linewidth=1, label='EV in home')
        axes[0].plot(tt, (Eev4_modified[tt] / C_ev) * 100, 'r', linewidth=1, label='EV out of Home')
        axes[0].legend()
        axes[0].set_ylabel(r'SOC$_{EV}$ (%)', fontsize=14)
        axes[0].grid(True, which='both')
        #axes[0].set_title(f'EV Results for {dd}-th and {dd + 1}-th day (weekends)')

        # ─────── SECOND SUBPLOT: Power Flows + Cbuy Line ───────
        # EV charging, discharging, travel bars
        axes[1].bar(tt, Pev_ch[tt], label='Pch', align='center', color='#1f77b4')  # blue
        axes[1].bar(tt, Pev_dch[tt], bottom=Pev_ch[tt], label='Pdch', align='center', color='#ff7f0e')  # orange
        axes[1].bar(tt, Pev_travel[tt], bottom=Pev_ch[tt] + Pev_dch[tt], label='Ptravel', align='center', color='#2ca02c')  # green

        if Grid > 0:
            # Secondary y-axis for grid price
            price_ax = axes[1].twinx()
            price_ax.set_ylabel('Grid Price [$/kWh]', fontsize=14, color='red')
            # Plot full Cbuy line
            price_ax.plot(tt, Cbuy[tt], color='red', linewidth=1.5, label='Cbuy', drawstyle='steps-pre')
            price_ax.set_ylim(0, np.max(Cbuy[tt]) * 1.05)  # Set y-axis for Cbuy

            # Highlight discharging periods with shading
            mask_dch = Pev_dch[tt] > 0  # boolean mask for discharging hours

            price_ax.fill_between(tt, Cbuy[tt], where=mask_dch, color='red', alpha=0.3, step='pre')
            price_ax.tick_params(axis='y', labelcolor='red', colors='red')

            # Combine legends
            h1, l1 = axes[1].get_legend_handles_labels()
            h2, l2 = price_ax.get_legend_handles_labels()
            price_ax.legend(h1 + h2, l1 + l2, loc='upper left', framealpha=1.0, facecolor='white', edgecolor='black', fontsize=10)
        else:
            axes[1].legend()

        # Labels and formatting
        axes[1].set_xlabel('Time [h]', fontsize=14)
        axes[1].set_ylabel('EV Power Flow [kW]', fontsize=14)
        max_power = np.max(Pev_ch[tt] + Pev_dch[tt] + Pev_travel[tt])
        axes[1].set_ylim(0, max_power * 1.2)  # Add 20% buffer
        axes[1].set_xticks(tt[::5])
        axes[1].set_xlim(tt[0], tt[-1])

        # ─────── Save the figure ───────
        plt.tight_layout()
        plt.savefig('output/figs/EV Sp Results 1.png', dpi=300)


        # Second figure
        # ─────── Inputs ───────
        dd = 10  # target day index (1-based)
        tt = np.arange(24 * (dd - 1) + 1, 24 * (dd + 1) + 1)  # 48-hour window

        fig, axes = plt.subplots(2, 1, figsize=(10, 6), sharex=True)

        # ─────── FIRST SUBPLOT: SOC (%) ───────
        Eev4_modified = Eev[0:8760].copy()
        EV_p_modified = EV_p.copy()

        # Mark the first hour when EV returns as 'still out of home'
        for t in range(len(EV_p) - 1):
            if EV_p[t] == 0 and EV_p[t + 1] == 1:
                EV_p_modified[t + 1] = 0

        # Set NaN for out-of-home values
        Eev4_modified[EV_p_modified == 1] = np.nan

        # Plot SOC in blue and red
        axes[0].plot(tt, (Eev[tt] / C_ev) * 100, 'b-.', linewidth=1, label='EV in home')
        axes[0].plot(tt, (Eev4_modified[tt] / C_ev) * 100, 'r', linewidth=1, label='EV out of Home')
        axes[0].legend()
        axes[0].set_ylabel(r'SOC$_{EV}$ (%)', fontsize=14)
        axes[0].grid(True, which='both')
        # axes[0].set_title(f'EV Results for {dd}-th and {dd + 1}-th day (weekends)')

        # ─────── SECOND SUBPLOT: Power Flows + Cbuy Line ───────
        # EV charging, discharging, travel bars
        axes[1].bar(tt, Pev_ch[tt], label='Pch', align='center', color='#1f77b4')  # blue
        axes[1].bar(tt, Pev_dch[tt], bottom=Pev_ch[tt], label='Pdch', align='center', color='#ff7f0e')  # orange
        axes[1].bar(tt, Pev_travel[tt], bottom=Pev_ch[tt] + Pev_dch[tt], label='Ptravel', align='center', color='#2ca02c')  # green

        if Grid > 0:
            # Secondary y-axis for grid price
            price_ax = axes[1].twinx()
            price_ax.set_ylabel('Grid Price [$/kWh]', fontsize=14, color='red')
            # Plot full Cbuy line
            price_ax.plot(tt, Cbuy[tt], color='red', linewidth=1.5, label='Cbuy', drawstyle='steps-pre')
            price_ax.set_ylim(0, np.max(Cbuy[tt]) * 1.05)  # Set y-axis for Cbuy

            # Highlight discharging periods with shading
            mask_dch = Pev_dch[tt] > 0  # boolean mask for discharging hours

            price_ax.fill_between(tt, Cbuy[tt], where=mask_dch, color='red', alpha=0.3, step='pre')
            price_ax.tick_params(axis='y', labelcolor='red', colors='red')

            # Combine legends
            h1, l1 = axes[1].get_legend_handles_labels()
            h2, l2 = price_ax.get_legend_handles_labels()
            price_ax.legend(h1 + h2, l1 + l2, loc='upper left', framealpha=1.0, facecolor='white', edgecolor='black', fontsize=10)

        else:
            axes[1].legend()

        # Labels and formatting
        axes[1].set_xlabel('Time [h]', fontsize=14)
        axes[1].set_ylabel('EV Power Flow [kW]', fontsize=14)
        max_power = np.max(Pev_ch[tt] + Pev_dch[tt] + Pev_travel[tt])
        axes[1].set_ylim(0, max_power * 1.2)  # Add 20% buffer
        axes[1].set_xticks(tt[::5])
        axes[1].set_xlim(tt[0], tt[-1])

        # ─────── Save the figure ───────
        plt.tight_layout()
        plt.savefig('output/figs/EV Sp Results 2.png', dpi=300)

