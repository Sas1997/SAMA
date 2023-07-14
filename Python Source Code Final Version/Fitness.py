from Input_Data import InData
import numpy as np
from numba import jit
from math import ceil
from EMS import EMS


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
Q_lifetime = InData.Q_lifetime
ef_bat = InData.ef_bat
b = InData.b
C_fuel = InData.C_fuel
R_DG = InData.R_DG
TL_DG = InData.TL_DG
MO_DG = InData.MO_DG
SOC_max = InData.SOC_max
SOC_min = InData.SOC_min
SOC_initial = InData.SOC_initial
n_I = InData.n_I
Grid = InData.Grid
Cbuy = InData.Cbuy
a = InData.a
LR_DG = InData.LR_DG
Pbuy_max = InData.Pbuy_max
Psell_max = InData.Psell_max
self_discharge_rate = InData.self_discharge_rate
alfa_battery = InData.alfa_battery
c = InData.c
k = InData.k
Imax = InData.Imax
Vnom = InData.Vnom
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


def fitness(X, final_solution=False, print_result=False):
    if X.size == 1:
        X = X[0]

    NT = Eload.size  # time step numbers
    Npv = X[0]  # PV number
    Nwt = X[1]  # WT number
    Nbat = X[2]  # Battery pack number
    N_DG = X[3]  # number of Diesel Generator
    Cn_I = X[4]  # Inverter Capacity

    Pn_PV = Npv * Ppv_r  # PV Total Capacity
    Pn_WT = Nwt * Pwt_r  # WT Total Capacity
    Cn_B = Nbat * Cbt_r  # Battery Total Capacity
    Pn_DG = N_DG * Cdg_r  # Diesel Total Capacity

    # PV Power Calculation
    #Tc = T + (((Tc_noct - 20) / 800) * G)  # Module Temprature
    # Module Temperature
    Tc = (T + (Tc_noct - Ta_noct) * (G / G_noct) * (1 - ((n_PV * (1 - Tcof * Tref)) / gama))) / (1 + (Tc_noct - Ta_noct) * (G / G_noct) * ((Tcof * n_PV) / gama))
    Ppv = fpv * Pn_PV * (G / Gref) * (1 + Tcof * (Tc - Tref))  # output power(kw)_hourly

    # Wind turbine Power Calculation
    v1 = Vw  # hourly wind speed
    v2 = ((h_hub / h0) ** (alfa_wind_turbine)) * v1  # v1 is the speed at a reference height;v2 is the speed at a hub height h2

    Pwt = np.zeros(8760)
    true_value = np.logical_and(v_cut_in <= v2, v2 < v_rated)
    Pwt[np.logical_and(v_cut_in <= v2, v2 < v_rated)] = v2[true_value] ** 3 * (Pwt_r / (v_rated ** 3 - v_cut_in ** 3)) - (v_cut_in ** 3 / (v_rated ** 3 - v_cut_in ** 3)) * (Pwt_r)
    Pwt[np.logical_and(v_rated <= v2, v2 < v_cut_out)] = Pwt_r
    Pwt = Pwt * Nwt

    ## Energy Management
    # Battery Wear Cost
    Cbw = R_B * Cn_B / (Nbat * Q_lifetime * np.sqrt(ef_bat)) if Cn_B > 0 else 0

    #  DG Fix cost
    cc_gen = b * Pn_DG * C_fuel + (R_DG * Pn_DG / TL_DG) + MO_DG

    Pdg, Ens, Pbuy, Psell, Edump, Pch, Pdch, Eb= EMS(Ppv, Pwt, Eload, Cn_B, Nbat, Pn_DG, NT,
                                                                    SOC_max, SOC_min, SOC_initial, n_I, Grid, Cbuy, a,
                                                                    Cn_I, LR_DG, C_fuel, Pbuy_max, Psell_max, cc_gen,
                                                                    Cbw,self_discharge_rate, alfa_battery, c, k, Imax, Vnom,
                                                                    ef_bat)

    q = (a * Pdg + b * Pn_DG) * (Pdg > 0)  # Fuel consumption of a diesel generator

    ## Installation and operation cost

    # Total Investment cost ($)
    I_Cost = C_PV * (1 - RE_incentives) * Pn_PV + C_WT * (1 - RE_incentives) * Pn_WT + C_DG * Pn_DG + C_B * (1 - RE_incentives) * Cn_B + C_I * (1 - RE_incentives) * Cn_I + C_CH * (1 - RE_incentives) + Engineering_Costs * (1 - RE_incentives) * Pn_PV

    Top_DG = np.sum(Pdg > 0) + 1
    L_DG = TL_DG / Top_DG
    RT_DG = ceil(n / L_DG) - 1  # Replecement time

    # Total Replacement Cost ($/year)

    RC_PV = np.zeros(n)
    RC_WT = np.zeros(n)
    RC_DG = np.zeros(n)
    RC_B = np.zeros(n)
    RC_I = np.zeros(n)
    RC_CH = np.zeros(n)

    RC_PV[np.arange(L_PV + 1, n, L_PV)] = R_PV * Pn_PV / (1 + ir) ** (np.arange(1.001 * L_PV, n, L_PV))
    RC_WT[np.arange(L_WT + 1, n, L_WT)] = R_WT * Pn_WT / (1 + ir) ** (np.arange(1.001 * L_WT, n, L_WT))
    RC_DG[np.arange(L_DG + 1, n, L_DG).astype(np.int32)] = R_DG * Pn_DG / (1 + ir) ** (np.arange(L_DG+1, n, L_DG))
    RC_B[np.arange(L_B + 1, n, L_B).astype(np.int32)] = R_B * Cn_B / (1 + ir) ** (np.arange(1.001 * L_B, n, L_B))
    RC_I[np.arange(L_I + 1, n, L_I)] = R_I * Cn_I / (1 + ir) ** (np.arange(1.001 * L_I, n, L_I))
    RC_CH[np.arange(L_CH + 1, n, L_CH)] = R_CH / (1 + ir) ** (np.arange(1.001 * L_CH, n, L_CH))
    R_Cost = RC_PV + RC_WT + RC_DG + RC_B + RC_I + RC_CH

    # Total M&O Cost ($/year)
    MO_Cost = (MO_PV * Pn_PV + MO_WT * Pn_WT + MO_DG * Pn_DG * np.sum(Pdg > 0) + MO_B * Cn_B + MO_I * Cn_I + MO_CH) / (1 + ir) ** np.array(range(1, n + 1))

    # DG fuel Cost
    C_Fu = np.sum(C_fuel * q) / (1 + ir) ** np.array(range(1, n + 1))

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
    Salvage = S_PV + S_WT + S_DG + S_B + S_I + S_CH

    # Emissions produced by Disesl generator (g)
    DG_Emissions = np.sum(q * (CO2 + NOx + SO2)) / 1000  # total emissions (kg/year)
    Grid_Emissions = np.sum(Pbuy * (E_CO2 + E_SO2 + E_NOx)) / 1000  # total emissions (kg/year)

    Grid_Cost = ((Annual_expenses + np.sum(Service_charge) + np.sum(Pbuy * Cbuy) - np.sum(Psell * Csell)) * 1 / (1 + ir) ** np.array(range(1, n+1))) * (1 + Grid_Tax) * (Grid > 0)

    # Capital recovery factor
    CRF = ir * (1 + ir) ** n / ((1 + ir) ** n - 1)

    # Totall Cost
    NPC = (((I_Cost + np.sum(R_Cost) + np.sum(MO_Cost) + np.sum(C_Fu) - Salvage) * (1 + System_Tax)) + np.sum(Grid_Cost))
    Operating_Cost = (CRF * (((np.sum(R_Cost) + np.sum(MO_Cost) + np.sum(C_Fu) - Salvage) * (1 + System_Tax)) + np.sum(Grid_Cost)))

    LCOE = CRF * NPC / np.sum(Eload - Ens + Psell)
    LEM = (DG_Emissions + Grid_Emissions) / np.sum(Eload - Ens)
    Ebmin = SOC_min * Cn_B
    Pb_min = (Eb - Ebmin) + Pdch
    Ptot = (Ppv + Pwt + Pb_min) * n_I + Pdg + Grid * Pbuy_max
    DE = np.maximum(Eload - Ptot, 0)

    LPSP = np.sum(Ens) / np.sum(Eload)
    RE = 1 - np.sum(Pdg + Pbuy) / np.sum(Eload + Psell-Ens)
    if (np.isnan(RE)):
        RE = 0

    Z = LCOE + EM * LEM + 1e6 * (LPSP > LPSP_max) + 1e6 * (RE < RE_min) + 100 * (I_Cost > Budget) +\
        1e8 * np.maximum(0, LPSP - LPSP_max) + 1e8 * np.maximum(0, RE_min - RE) + 1e4 * np.maximum(0, I_Cost - Budget)
    return Z



