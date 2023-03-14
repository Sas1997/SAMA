import numpy as np
from EMS import energy_management
from InputData import InData

def fitness(X):
    if (len(X)) == 1:
        X = X[0]

    Npv = round(X[0])  # PV number
    Nwt = round(X[1])  # WT number
    Nbat = round(X[2])  # Battery pack number
    N_DG = round(X[3])  # number of Diesel Generator
    Cn_I = X[4]  # Inverter Capacity

    Pn_PV = Npv * InData.Ppv_r  # PV Total Capacity
    Pn_WT = Nwt * InData.Pwt_r  # WT Total Capacity
    Cn_B = Nbat * InData.Cbt_r  # Battery Total Capacity
    Pn_DG = N_DG * InData.Cdg_r  # Diesel Total Capacity

    # %% PV Power Calculation

    Ppv = np.multiply(InData.fpv * Pn_PV * (InData.G / InData.Gref),
                      (1 + InData.Tcof * (InData.Tc - InData.Tref)))  # output power(kw) _hourly

    # %% Wind turbine Power Calculation
    v1 = InData.Vw  # hourly wind speed
    v2 = ((InData.h_hub / InData.h0) ** (
        InData.alfa_wind_turbine)) * v1  # v1 is the speed at a reference heightv2 is the speed at a hub height h2
    Pwt = np.zeros(InData.NT)

    Pwt[v2 < InData.v_cut_in] = 0
    Pwt[v2 > InData.v_cut_out] = 0
    true_value = np.logical_and(InData.v_cut_in <= v2, v2 < InData.v_rated)
    Pwt[np.logical_and(InData.v_cut_in <= v2, v2 < InData.v_rated)] = v2[true_value] ** 3 * (
            InData.Pwt_r / (InData.v_rated ** 3 - InData.v_cut_in ** 3)) - (InData.v_cut_in ** 3 / (
            InData.v_rated ** 3 - InData.v_cut_in ** 3)) * (InData.Pwt_r)
    Pwt[np.logical_and(InData.v_rated <= v2, v2 < InData.v_cut_out)] = InData.Pwt_r
    Pwt = Pwt * Nwt

    # %% Energy Management 
    # % Battery Wear Cost
    if Cn_B > 0:
        Cbw = InData.R_B * Cn_B / (Nbat * InData.Q_lifetime * np.sqrt(InData.ef_bat))
    else:
        Cbw = 0

    #  DG Fix cost
    cc_gen = InData.b * Pn_DG * InData.C_fuel + InData.R_DG * Pn_DG / InData.TL_DG + InData.MO_DG

    (Eb, Pdg, Edump, Ens, Pch, Pdch, Pbuy, Psell, Pinv) = energy_management(Ppv, Pwt, Cn_B, Nbat, Pn_DG, Cn_I, cc_gen,
                                                                          Cbw, InData)

    Eb=np.array(Eb)
    Pdg=np.array(Pdg)
    Edump=np.array(Edump)
    Ens=np.array(Ens)
    Pch=np.array(Pch)
    Pdch=np.array(Pdch)
    Pbuy=np.array(Pbuy)
    Psell=np.array(Psell)
    Pinv=np.array(Pinv)

    q = (InData.a * Pdg + InData.b * Pn_DG) * (Pdg > 0)  # Fuel consumption of a diesel generator

    # %% installation and operation cost

    # Total Investment cost ($)
    I_Cost = InData.C_PV * (1 - InData.RE_incentives) * Pn_PV + InData.C_WT * (1 - InData.RE_incentives) * Pn_WT + \
             InData.C_DG * Pn_DG + InData.C_B * (1 - InData.RE_incentives) * Cn_B + \
             InData.C_I * (1 - InData.RE_incentives) * Cn_I + InData.C_CH * (1 - InData.RE_incentives) + \
             InData.Engineering_Costs * (1 - InData.RE_incentives) * Pn_PV

    Top_DG = sum(Pdg > 0) + 1
    L_DG = np.round(InData.TL_DG / Top_DG)
    RT_DG = np.ceil(InData.n / L_DG) - 1  # Replecement time

    # Total Replacement cost ($)
    R_Cost = 0
    for i in range(InData.n):
        # adding RC_PV to R_Cost
        if 0 <= (i+1) % InData.L_PV < 1:
            R_Cost += InData.R_PV * Pn_PV / (1 + InData.ir) ** (1.001 * (i + 1))

        # adding RC_WT to R_Cost
        if 0 <= (i+1) % InData.L_WT < 1:
            R_Cost += InData.R_WT * Pn_WT / (1 + InData.ir) ** (1.001 * (i + 1))

        # adding RC_DG to R_Cost
        if 0 <= (i+1) % L_DG < 1:
            R_Cost += InData.R_DG * Pn_DG / (1 + InData.ir) ** (1.001 * (i + 1))

        # adding RC_B to R_Cost
        if 0 <= (i+1) % InData.L_B < 1:
            R_Cost += InData.R_B * Cn_B / (1 + InData.ir) ** (1.001 * (i + 1))

        # adding RC_I to R_Cost
        if 0 <= (i+1) % InData.L_I < 1:
            R_Cost += InData.R_I * Cn_I / (1 + InData.ir) ** (1.001 * (i + 1))

        # adding RC_CH to R_Cost
        if 0 <= (i+1) % InData.L_CH < 1:
            R_Cost += InData.R_CH / (1 + InData.ir) ** (1.001 * (i + 1))

    # Total M&O Cost ($/year)
    MO_Cost = (InData.MO_PV * Pn_PV + InData.MO_WT * Pn_WT + InData.MO_DG * np.sum(Pn_DG > 0) + InData.MO_B * Cn_B +
               InData.MO_I * Cn_I + InData.MO_CH) / (1 + InData.ir) ** np.array(range(1, InData.n + 1))

    # DG fuel Cost
    C_Fu = sum(InData.C_fuel * q) / (1 + InData.ir) ** np.array(range(1, InData.n + 1))

    # Salvage
    L_rem = (InData.RT_PV + 1) * InData.L_PV - InData.n
    S_PV = (InData.R_PV * Pn_PV) * L_rem / InData.L_PV * 1 / (1 + InData.ir) ** InData.n  # PV
    L_rem = (InData.RT_WT + 1) * InData.L_WT - InData.n
    S_WT = (InData.R_WT * Pn_WT) * L_rem / InData.L_WT * 1 / (1 + InData.ir) ** InData.n  # WT
    L_rem = (RT_DG + 1) * L_DG - InData.n
    S_DG = (InData.R_DG * Pn_DG) * L_rem / L_DG * 1 / (1 + InData.ir) ** InData.n  # DG
    L_rem = (InData.RT_B + 1) * InData.L_B - InData.n
    S_B = (InData.R_B * Cn_B) * L_rem / InData.L_B * 1 / (1 + InData.ir) ** InData.n
    L_rem = (InData.RT_I + 1) * InData.L_I - InData.n
    S_I = (InData.R_I * Cn_I) * L_rem / InData.L_I * 1 / (1 + InData.ir) ** InData.n
    L_rem = (InData.RT_CH + 1) * InData.L_CH - InData.n
    S_CH = (InData.R_CH) * L_rem / InData.L_CH * 1 / (1 + InData.ir) ** InData.n
    Salvage = S_PV + S_WT + S_DG + S_B + S_I + S_CH

    # Emissions produced by Disesl generator (g)
    DG_Emissions = sum(q * (InData.CO2 + InData.NOx + InData.SO2)) / 1000  # total emissions (kg/year)
    Grid_Emissions = sum(Pbuy * (InData.E_CO2 + InData.E_SO2 + InData.E_NOx)) / 1000  # total emissions (kg/year)

    Grid_Cost = (sum(Pbuy * InData.Cbuy) - sum(Psell * InData.Csell)) * 1 / (1 + InData.ir) ** np.array(
        range(1, InData.n + 1))

    # Capital recovery factor
    CRF = InData.ir * (1 + InData.ir) ** InData.n / ((1 + InData.ir) ** InData.n - 1)

    # Totall Cost
    NPC = I_Cost + R_Cost + sum(MO_Cost) + sum(C_Fu) - Salvage + sum(Grid_Cost) * (
            1 + InData.System_Tax)

    # Operating_Cost = CRF * (R_Cost + sum(MO_Cost) + sum(C_Fu) - Salvage + sum(Grid_Cost))
    if sum(InData.Eload - Ens) > 1:
        LCOE = CRF * NPC / sum(InData.Eload - Ens + Psell)  # Levelized Cost of Energy ($/kWh)
        LEM = (DG_Emissions + Grid_Emissions) / sum(InData.Eload - Ens)  # Levelized Emissions(kg/kWh)
    else:
        LCOE = 100
        LEM = 100

    LPSP = sum(Ens) / sum(InData.Eload)

    RE = 1 - sum(Pdg + Pbuy) / (sum(InData.Eload + Psell - Ens)+0.0000001)
    if (np.isnan(RE)):
        RE = 0

    Z = LCOE + InData.EM * LEM + 10 * (LPSP > InData.LPSP_max) + 10 * (RE < InData.RE_min) + 100 * (
            I_Cost > InData.Budget) + \
        100 * max(0, LPSP - InData.LPSP_max) + 100 * max(0, InData.RE_min - RE) + 100 * max(0, I_Cost - InData.Budget)

    return Z
