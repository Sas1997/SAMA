from Input_Data import InData
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
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
WT=InData.WT
daysInMonth=InData.daysInMonth

def Gen_Results(X):
    if (len(X)) == 1:
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
    Tc = (T + (Tc_noct - Ta_noct) * (G / G_noct) * (1 - ((n_PV * (1 - Tcof * Tref)) / gama))) / (1 + (Tc_noct - Ta_noct) * (G / G_noct) * ((Tcof * n_PV) / gama))
    Ppv = fpv * Pn_PV * (G / Gref) * (1 + Tcof * (Tc - Tref))  # output power(kw)_hourly

    # Wind turbine Power Calculation
    v1 = Vw  # hourly wind speed
    v2 = ((h_hub / h0) ** (
        alfa_wind_turbine)) * v1  # v1 is the speed at a reference height;v2 is the speed at a hub height h2

    Pwt = np.zeros(8760)
    true_value = np.logical_and(v_cut_in <= v2, v2 < v_rated)
    Pwt[np.logical_and(v_cut_in <= v2, v2 < v_rated)] = v2[true_value] ** 3 * (
            Pwt_r / (v_rated ** 3 - v_cut_in ** 3)) - (v_cut_in ** 3 / (v_rated ** 3 - v_cut_in ** 3)) * (Pwt_r)
    Pwt[np.logical_and(v_rated <= v2, v2 < v_cut_out)] = Pwt_r
    Pwt = Pwt * Nwt

    ## Energy Management
    # Battery Wear Cost
    Cbw = R_B * Cn_B / (Nbat * Q_lifetime * np.sqrt(ef_bat)) if Cn_B > 0 else 0

    #  DG Fix cost
    cc_gen = b * Pn_DG * C_fuel + ((R_DG * Pn_DG) / (TL_DG)) + MO_DG

    Pdg, Ens, Pbuy, Psell, Edump, Pch, Pdch, Eb = EMS(Ppv, Pwt, Eload, Cn_B, Nbat, Pn_DG, NT,
                                                      SOC_max, SOC_min, SOC_initial, n_I, Grid, Cbuy, a,
                                                      Cn_I, LR_DG, C_fuel, Pbuy_max, Psell_max, cc_gen,
                                                      Cbw, self_discharge_rate, alfa_battery, c, k, Imax, Vnom,
                                                      ef_bat)

    q = (a * Pdg + b * Pn_DG) * (Pdg > 0)  # Fuel consumption of a diesel generator

    ## Installation and operation cost

    # Total Investment cost ($)
    I_Cost = C_PV * (1 - RE_incentives) * Pn_PV + C_WT * (1 - RE_incentives) * Pn_WT + C_DG * Pn_DG + C_B * (1 - RE_incentives) * Cn_B + C_I * (1 - RE_incentives) * Cn_I + C_CH * (1 - RE_incentives)*(Nbat > 0) + Engineering_Costs * (1 - RE_incentives) * Pn_PV
    I_Cost_without_incentives = C_PV * Pn_PV + C_WT * Pn_WT + C_DG * Pn_DG + C_B * Cn_B + C_I * Cn_I + C_CH*(Nbat > 0) + Engineering_Costs * Pn_PV
    Total_incentives_received = I_Cost_without_incentives - I_Cost


    Top_DG = np.sum(Pdg > 0) + 1
    L_DG = TL_DG / Top_DG
    RT_DG = ceil(n / L_DG) - 1  # Replecement time

    # Total Replacement cost ($)
    RC_PV = np.zeros(n)
    RC_WT = np.zeros(n)
    RC_DG = np.zeros(n)
    RC_B = np.zeros(n)
    RC_I = np.zeros(n)
    RC_CH = np.zeros(n)

    RC_PV[np.arange(L_PV + 1, n, L_PV)] = R_PV * Pn_PV / (1 + ir) ** (np.arange(1.001 * L_PV, n, L_PV))
    RC_WT[np.arange(L_WT + 1, n, L_WT)] = R_WT * Pn_WT / (1 + ir) ** (np.arange(1.001 * L_WT, n, L_WT))
    RC_DG[np.arange(L_DG + 1, n, L_DG).astype(np.int32)] = R_DG * Pn_DG / (1 + ir) ** (np.arange(L_DG + 1, n, L_DG))
    RC_B[np.arange(L_B + 1, n, L_B).astype(np.int32)] = R_B * Cn_B / (1 + ir) ** (np.arange(1.001 * L_B, n, L_B))
    RC_I[np.arange(L_I + 1, n, L_I)] = R_I * Cn_I / (1 + ir) ** (np.arange(1.001 * L_I, n, L_I))
    RC_CH[np.arange(L_CH + 1, n, L_CH)] = R_CH / (1 + ir) ** (np.arange(1.001 * L_CH, n, L_CH))
    R_Cost = RC_PV + RC_WT + RC_DG + RC_B + RC_I + RC_CH*(Nbat > 0)

    # Total M&O Cost ($/year)
    MO_Cost = (MO_PV * Pn_PV + MO_WT * Pn_WT + MO_DG * Pn_DG * np.sum(Pdg > 0) + MO_B * Cn_B + MO_I * Cn_I + MO_CH*(Nbat > 0)) / (1 + ir) ** np.array(range(1, n + 1))

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
    Salvage = S_PV + S_WT + S_DG + S_B + S_I + S_CH*(Nbat > 0)

    # Emissions produced by Disesl generator (g)
    DG_Emissions = np.sum(q * (CO2 + NOx + SO2)) / 1000  # total emissions (kg/year)
    Grid_Emissions = np.sum(Pbuy * (E_CO2 + E_SO2 + E_NOx)) / 1000  # total emissions (kg/year)

    Grid_Cost = ((Annual_expenses + np.sum(Service_charge) + np.sum(Pbuy * Cbuy) - np.sum(Psell * Csell)) * 1 / (1 + ir) ** np.array(range(1, n + 1))) * (1 + Grid_Tax) * (Grid > 0)
    Grid_Cost_onlyG = ((Annual_expenses + np.sum(Service_charge) + np.sum(Eload * Cbuy)) * 1 / (1 + ir) ** np.arange(1, n + 1)) * (1 + Grid_Tax)

    # Capital recovery factor
    CRF = ir * (1 + ir) ** n / ((1 + ir) ** n - 1)

    # Total Cost
    NPC = (((I_Cost + np.sum(R_Cost) + np.sum(MO_Cost) + np.sum(C_Fu) - Salvage) * (1 + System_Tax)) + np.sum(Grid_Cost))
    NPC_without_incentives = (((I_Cost_without_incentives + np.sum(R_Cost) + np.sum(MO_Cost) + np.sum(C_Fu) - Salvage) * (1 + System_Tax)) + np.sum(Grid_Cost))
    NPC_Grid = np.sum(Grid_Cost_onlyG)
    Operating_Cost = CRF * (((np.sum(R_Cost) + np.sum(MO_Cost) + np.sum(C_Fu) - Salvage) * (1 + System_Tax)) + np.sum(Grid_Cost))

    LCOE = CRF * NPC / np.sum(Eload - Ens + Psell) # Levelized Cost of Energy ($/kWh)
    LCOE_without_incentives = CRF * NPC_without_incentives / np.sum(Eload - Ens + Psell)
    LCOE_Grid = CRF * NPC_Grid / np.sum(Eload - Ens)
    LEM = (DG_Emissions + Grid_Emissions) / np.sum(Eload - Ens)  # Levelized Emissions(kg/kWh)

    Ebmin = SOC_min * Cn_B  # Battery minimum energy
    Pb_min= (Eb - Ebmin) + Pdch  # Battery minimum power in t=2,3,...,NT
    Ptot = (Ppv + Pwt + Pb_min) * n_I + Pdg + Grid * Pbuy_max  # total available power in system for each hour
    DE = np.maximum((Eload - Ptot), 0)  # power shortage in each hour
    LPSP = np.sum(Ens) / np.sum(Eload)
    RE = 1 - np.sum(Pdg + Pbuy) / np.sum(Eload + Psell - Ens)  # sum(Ppv + Pwt - Edump) / sum(Eload + Psell - Ens)
    if (np.isnan(RE)):
        RE = 0

    print(LCOE)

    import pandas as pd

    #Extracting data for plotting
    data = {'Ppv': Ppv, 'Pdg': Pdg, 'Pch': Pch, 'Pdch': Pdch, 'SOC': Eb / Cn_B}

    df = pd.DataFrame(data)

    df.to_csv('Outputforplotting.csv', index=False)


    print(' ')
    print('System Size ')
    print('Cpv  (kW) =', round(Pn_PV, 2))
    if WT == 1:
        print('Cwt  (kW) =', round(Pn_WT, 2))
    print('Cbat (kWh) =', round(Cn_B))
    print('Cdg  (kW) =', round(Pn_DG, 2))
    print('Cinverter (kW) =', round(Cn_I, 2))

    print(' ')
    print('Result: ')
    print('NPC  = $', round(NPC, 2))
    print('NPC without incentives = $', round(NPC_without_incentives, 2))
    print('NPC for only Grid connected system = $', round(NPC_Grid, 2))
    print('LCOE  =', round(LCOE, 2), '$/kWh')
    print('LCOE without incentives =', round(LCOE_without_incentives, 2), '$/kWh')
    print('LCOE for only Grid connected system =', round(LCOE_Grid, 2), '$/kWh')
    print('Operating Cost  = $', round(Operating_Cost, 2))
    print('Initial Cost  = $', round(I_Cost, 2))
    print('Initial Cost without incentives= $', round(I_Cost_without_incentives, 2))
    print('Total incentives received= $', round(Total_incentives_received, 2))
    print('RE  =', round(100 * RE, 2), '%')
    print('Total operation and maintenance cost  = $', round(np.sum(MO_Cost), 2))

    print(' ')
    print('PV Power  =', np.sum(Ppv), 'kWh')
    if WT == 1:
        print('WT Power  =', np.sum(Pwt), 'kWh')
    print('DG Power  =', np.sum(Pdg), 'kWh')
    print('Battery Energy In  =', np.sum(Pch), 'kWh')
    print('Battery Energy Out  =', np.sum(Pdch), 'kWh')
    print('LPSP  =', round(100 * LPSP, 2), '%')
    print('Excess Electricity =', np.sum(Edump), 'kWh')

    if Grid == 1:
        Total_Pbuy = (np.sum(Pbuy) * n) * (Grid > 0)
        Total_Psell = (np.sum(Psell) * n) * (Grid > 0)
        print('Total power bought from Grid= ', Total_Pbuy, 'kWh')
        print('Power sold to Grid= ', Total_Psell, 'kWh')
        print('Total Money paid to the Grid= $', round(np.sum(Grid_Cost), 2))
        print('Grid Emissions   =', Grid_Emissions, '(kg/year)')

    print('Total Money paid by the user= $', round(np.sum(NPC), 2))
    print('total fuel consumed by DG   =', np.sum(q), '(Liter/year)')
    print('DG Emissions   =', DG_Emissions, '(kg/year)')
    print('LEM  =', LEM, 'kg/kWh')

    Investment = np.zeros(n)
    Investment[0] = I_Cost
    Salvage1 = np.zeros(n)
    Salvage1[n - 1] = Salvage
    Salvage1[0] = 0
    Salvage = Salvage1
    Operating = np.zeros(n)
    Operating[0:n + 1] = MO_PV * Pn_PV + MO_WT * Pn_WT + MO_DG \
                         * Pn_DG + MO_B * Cn_B + MO_I * Cn_I + sum(Pbuy * Cbuy) - sum(Psell * Csell)
    Fuel = np.zeros(n)
    Fuel[0:n + 1] = sum(C_fuel * q)

    RC_PV[np.arange(L_PV + 1, n, L_PV)] = R_PV * Pn_PV
    RC_WT[np.arange(L_WT + 1, n, L_WT)] = R_WT * Pn_WT
    RC_DG[np.arange(L_DG + 1, n, L_DG).astype(np.int32)] = R_DG * Pn_DG
    RC_B[np.arange(L_B + 1, n, L_B).astype(np.int32)] = R_B * Cn_B
    RC_I[np.arange(L_I + 1, n, L_I)] = R_I * Cn_I
    Replacement = RC_PV + RC_WT + RC_DG + RC_B + RC_I

    Cash_Flow = np.zeros((len(Investment), 5))
    Cash_Flow[:, 0] = -Investment
    Cash_Flow[:, 1] = -Operating
    Cash_Flow[:, 2] = Salvage
    Cash_Flow[:, 3] = -Fuel
    Cash_Flow[:, 4] = -Replacement

    # Set the font to Times New Roman
    plt.rcParams["font.family"] = "Times New Roman"

    plt.figure()
    for kk in range(5):
        plt.bar(range(0, 25), Cash_Flow[:, kk])
    plt.legend(['Capital', 'Operating', 'Salvage', 'Fuel', 'Replacement'])
    plt.title('Cash Flow')
    plt.xlabel('Year')
    plt.ylabel('$')

    #Grid purchase and sale figure
    if Grid == 0:
        Pbuy = 0
        Psell = 0
    else:
        fig, ax = plt.subplots(figsize=(14, 10), dpi=300)  # Increased figure size and resolution
        ax.plot(Pbuy, marker='.', linestyle='-', linewidth=0.5, color='blue',label='Buy')  # Added markers and reduced line width
        ax.plot(Psell, marker='.', linestyle='-', linewidth=0.5, color='red',label='Sell')  # Added markers and reduced line width
        ax.set_ylabel('Pgrid (kWh)', fontsize=22)  # Increased font size
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
        ax.set_xlim([0, max(hours_per_month)])
        ax.set_ylim([0, max(max(Pbuy), max(Psell))])
        # Adjust the margins and space between subplots
        plt.subplots_adjust(left=0.05, right=0.85, top=0.95, bottom=0.08)

    # Energy Distribution figure
    fig, ax = plt.subplots(figsize=(14, 10), dpi=300)  # Increased figure size and resolution
    # Increased linewidth and added distinct colors, line styles, and markers for visibility
    ax.plot(Eload - Ens, linestyle='-', linewidth=2, color='blue', marker='o', markersize=4, label='Load-Ens')
    ax.plot(Pdg, linestyle='--', linewidth=2, color='red', marker='x', markersize=4, label='Pdg')
    ax.plot(Pch - Pdch, linestyle='-.', linewidth=2, color='green', marker='^', markersize=4, label='Pbat')
    ax.plot(Ppv + Pwt, linestyle=':', linewidth=2, color='purple', marker='s', markersize=4, label='P_RE')
    ax.set_ylabel('Power [kWh]', fontsize=22)  # Increased font size
    ax.set_xlabel('Month', fontsize=22)  # Increased font size
    ax.tick_params(axis='both', which='major', labelsize=18)  # Increased tick label size
    ax.legend(loc='upper left', bbox_to_anchor=(1, 1), fontsize=22)  # Moved legend closer to the figure
    ax.set_title('Energy Distribution', fontsize=22)  # Increased font size
    # Set x-ticks
    hours_per_month = [0, 31 * 24, 59 * 24, 90 * 24, 120 * 24, 151 * 24, 181 * 24, 212 * 24, 243 * 24, 273 * 24, 304 * 24, 334 * 24]
    month_labels = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
    ax.set_xticks(hours_per_month)
    ax.set_xticklabels(month_labels)
    # Set x-axis to start from zero and y-axis according to data
    ax.set_xlim([0, max(hours_per_month)])
    ax.set_ylim([0, max(max(Eload - Ens), max(Pdg), max(Pch - Pdch), max(Ppv + Pwt))])
    # Adjust the margins and space between subplots
    plt.subplots_adjust(left=0.05, right=0.82, top=0.95, bottom=0.08)

    # State of charge figure
    if Nbat > 0:
        fig, ax = plt.subplots(figsize=(14, 10), dpi=300)  # Increased figure size and resolution
        # Plot 'State of Charge'
        ax.plot(Eb / Cn_B, marker='.', linestyle='-', linewidth=0.5, color='blue')
        ax.set_ylabel('SOC', fontsize=22)  # Increased font size
        ax.set_xlabel('Month', fontsize=22)  # Increased font size
        ax.tick_params(axis='both', which='major', labelsize=18)  # Increased tick label size
        ax.set_title('State of Charge', fontsize=22)  # Increased font size
        # Assuming a non-leap year (365 days), set the x-ticks at the beginning of each month.
        hours_per_month = [0, 31 * 24, 59 * 24, 90 * 24, 120 * 24, 151 * 24, 181 * 24, 212 * 24, 243 * 24, 273 * 24, 304 * 24, 334 * 24]
        month_labels = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
        ax.set_xticks(hours_per_month)
        ax.set_xticklabels(month_labels)
        # Set x-axis and y-axis to start from zero
        ax.set_xlim([0, max(hours_per_month)])
        ax.set_ylim([0, max(Eb / Cn_B)])
        # Adjust the margins and space between subplots
        plt.subplots_adjust(left=0.05, right=0.95, top=0.95, bottom=0.08)

    # Plot results for one specific day
    # Function to filter out data series with sum less than 0.1 in the specified range
    def non_zero_data_series(data_series, t1, t2):
        return [(title, data, label) for title, data, label in data_series if np.sum(data[t1:t2 + 1]) >= 0.1]

    # Specific day number
    Day = 180
    t1 = Day * 24
    t2 = (Day + 1) * 24

    data_series = [('Load Profile', Eload, '$E_{load}$ [kWh]'),
                   ('Plane of Array Irradiance', G, 'G[W/m^2]'),
                   ('Ambient Temperature', T, '$T$[^o C]'),
                   ('PV Power', Ppv, '$P_{pv}$ [kWh]'),
                   ('WT Energy', Pwt, '$P_{wt}$ [kWh]'),
                   ('Diesel Generator Energy', Pdg, '$E_{DG}$ [kWh]'),
                   ('Battery Energy Level', Eb, '$E_{b}$ [kWh]'),
                   ('State of Charge', Eb / Cn_B if not np.all(Eb[t1:t2] == 0) else np.zeros_like(Eb), 'SOC'),
                   ('Loss of Power Supply', Ens, 'LPS[kWh]'),
                   ('Dumped Energy', Edump, '$E_{dump}$ [kWh]'),
                   ('Battery decharge Energy', Pdch, '$E_{dch}$ [kWh]'),
                   ('Battery charge Energy', Pch, '$E_{ch}$ [kWh]')]

    # Apply the filter function to remove series with sum less than 0.1
    non_zero_series = non_zero_data_series(data_series, t1, t2)

    n_non_zeros = len(non_zero_series)  # Number of non-zero series
    n_cols = 3  # Number of columns
    n_rows = n_non_zeros // n_cols + (n_non_zeros % n_cols > 0)  # Number of rows

    fig, axs = plt.subplots(n_rows, n_cols, figsize=(16, n_rows * 4))  # Adjusting figure size based on number of rows
    fig.suptitle('Results for ' + str(Day) + ' -th day', fontsize=24)  # Increase font size
    axs = axs.flatten()  # Flatten the 2D array to 1D for easy iteration

    for i, (title, data, label) in enumerate(non_zero_series):
        ax = axs[i]
        ax.plot(range(t1, t2), data[t1:t2], linewidth=2)  # Increase line width
        ax.set_title(title, fontsize=12)  # Increase font size
        ax.set_ylabel(label, fontsize=10)  # Increase font size
        ax.set_xlabel('t[hour]', fontsize=10)  # Increase font size
        ax.tick_params(axis='both', which='major', labelsize=10)  # Increase font size

    for j in range(i + 1, n_rows * n_cols):  # Turn off unused subplots
        axs[j].axis('off')

    plt.tight_layout(rect=[0, 0, 1, 0.96])  # Adjust the layout to make space for the title

    # Utility figures

    #The figure showing each day average cost of energy system
    A_l = np.zeros((12, 31))
    index = 1
    for m in range(12):
        index1 = index
        for d in range(daysInMonth[m]):
            Total_daily_load = np.sum(Eload[index1:index1 + 23])
            A_l[m, d] = Total_daily_load
            index1 = index1 + 24
        index = (24 * daysInMonth[m]) + index

    AE_c = np.round(LCOE * A_l, 2)

    # Plotting heat map of each day average cost of energy system in each month
    AE_c[np.where(AE_c == 0)] = np.nan
    # Increase the figure size
    fig = plt.figure(figsize=(20, 15))
    fig.suptitle('Average cost of energy system for each day ($)', fontsize=24, y=0.92)
    # Define grid and increase the space between heatmap and colorbar
    gs = gridspec.GridSpec(2, 1, height_ratios=[19, 1])
    gs.update(wspace=0.025, hspace=0.2)  # Increase hspace
    ax0 = plt.subplot(gs[0])
    # Increase the size of the annotation and set linewidths to 0
    sns.heatmap(AE_c, cmap='jet', annot=True, fmt=".1f", yticklabels=False, ax=ax0, cbar=False, annot_kws={"size": 18},
                linewidths=0)
    # Set the y-tick labels
    months = ["January", "February", "March", "April", "May", "June", "July", "August", "September", "October", "November", "December"]
    ax0.set_yticks(np.arange(len(months)) + 0.5)  # Centering the labels
    ax0.set_yticklabels(months)
    # Adjust the x-ticks and labels to start from 1
    xticks = np.arange(1, AE_c.shape[1] + 1)
    ax0.set_xticks(xticks - 0.5, minor=False)  # Shift the ticks to be centered
    ax0.set_xticklabels(xticks)
    # Increase the size of y and x tick labels
    ax0.tick_params(axis='y', labelsize=22)
    ax0.tick_params(axis='x', labelsize=22)
    # Define colorbar axes and plot colorbar
    ax1 = plt.subplot(gs[1])
    cb = plt.colorbar(ax0.collections[0], cax=ax1, orientation='horizontal')
    # Increase the size of colorbar tick labels
    ax1.tick_params(labelsize=22)

    # Calculate average hourly grid cost for each day in each month
    Gh_c = np.zeros((12, 31))
    index = 1
    for m in range(12):
        index1 = index
        for d in range(daysInMonth[m]):
            gridcost = np.mean(Cbuy[index1:index1 + 23])
            Gh_c[m, d] = gridcost
            index1 = index1 + 24
        index = (24 * daysInMonth[m]) + index

    # Plot average hourly Grid cost (Cbuy) for each day in each month heatmap
    Gh_c[np.where(Gh_c == 0)] = np.nan
    # Increase the figure size
    fig = plt.figure(7, figsize=(20, 15))
    fig.suptitle('Average Grid hourly cost in each day [$/kWh]', fontsize=24)
    # Define grid and increase the space between heatmap and colorbar
    gs = gridspec.GridSpec(2, 1, height_ratios=[19, 1])
    gs.update(wspace=0.025, hspace=0.2)  # Increase hspace
    ax0 = plt.subplot(gs[0])
    # Increase the size of the annotation and set linewidths to 0
    sns.heatmap(np.round(Gh_c, 2), cmap='jet', annot=True, fmt=".2f", yticklabels=False, ax=ax0, cbar=False, annot_kws={"size": 15}, linewidths=0)
    # Set the y-tick labels
    months = ["January", "February", "March", "April", "May", "June", "July", "August", "September", "October", "November", "December"]
    ax0.set_yticks(np.arange(len(months)) + 0.5)  # Centering the labels
    ax0.set_yticklabels(months)
    # Adjust the x-ticks and labels to start from 1
    xticks = np.arange(1, Gh_c.shape[1] + 1)
    ax0.set_xticks(xticks - 0.5, minor=False)  # Shift the ticks to be centered
    ax0.set_xticklabels(xticks)
    # Increase the size of y and x tick labels
    ax0.tick_params(axis='y', labelsize=22)
    ax0.tick_params(axis='x', labelsize=22)
    # Define colorbar axes and plot colorbar
    ax1 = plt.subplot(gs[1])
    cb = plt.colorbar(ax0.collections[0], cax=ax1, orientation='horizontal')
    # Increase the size of colorbar tick labels
    ax1.tick_params(labelsize=22)


    # Calculate average only grid connected system cost for each day in each month
    AG_c = np.round(LCOE_Grid * A_l, 2)

    # Plot average only grid connected system cost heatmap for each day in each month
    AG_c[np.where(AG_c == 0)] = np.nan
    # Increase the figure size
    fig = plt.figure(8, figsize=(20, 15))
    fig.suptitle('Average only Grid-connected system cost in each day [$]', fontsize=24)
    # Define grid and increase the space between heatmap and colorbar
    gs = gridspec.GridSpec(2, 1, height_ratios=[19, 1])
    gs.update(wspace=0.025, hspace=0.2)  # Increase hspace
    ax0 = plt.subplot(gs[0])

    # Increase the size of the annotation and set linewidths to 0
    sns.heatmap(AG_c, cmap='jet', annot=True, fmt=".1f", yticklabels=False, ax=ax0, cbar=False, annot_kws={"size": 15}, linewidths=0)
    # Set the y-tick labels
    months = ["January", "February", "March", "April", "May", "June", "July", "August", "September", "October", "November", "December"]
    ax0.set_yticks(np.arange(len(months)) + 0.5)  # Centering the labels
    ax0.set_yticklabels(months)
    # Adjust the x-ticks and labels to start from 1
    xticks = np.arange(1, AG_c.shape[1] + 1)
    ax0.set_xticks(xticks - 0.5, minor=False)  # Shift the ticks to be centered
    ax0.set_xticklabels(xticks)
    # Increase the size of y and x tick labels
    ax0.tick_params(axis='y', labelsize=22)
    ax0.tick_params(axis='x', labelsize=22)
    # Define colorbar axes and plot colorbar
    ax1 = plt.subplot(gs[1])
    cb = plt.colorbar(ax0.collections[0], cax=ax1, orientation='horizontal')
    # Increase the size of colorbar tick labels
    ax1.tick_params(labelsize=22)

    # Hourly Grid electrcity price color map
    # Assuming Cbuy is a 1D numpy array
    Cbuy_2D = np.reshape(Cbuy, (1, len(Cbuy)))  # Reshape to 2D
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

    # Calculate average money earned by selling electricity to grid in each day in each month
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
        AG_sc = np.round(Csell * AG_s, 2)

        # Plot average money earned by selling electricity to grid in each day in each month heatmap
        AG_sc[np.where(AG_sc == 0)] = np.nan
        # Increase the figure size
        fig = plt.figure(10, figsize=(20, 15))
        fig.suptitle('Sell to the Grid in each day [$/kWh]', fontsize=24)
        # Define grid and increase the space between heatmap and colorbar
        gs = gridspec.GridSpec(2, 1, height_ratios=[19, 1])
        gs.update(wspace=0.025, hspace=0.2)  # Increase hspace
        ax0 = plt.subplot(gs[0])
        # Increase the size of the annotation and set linewidths to 0
        sns.heatmap(AG_sc, cmap='jet', annot=True, fmt=".1f", yticklabels=False, ax=ax0, cbar=False, annot_kws={"size": 15}, linewidths=0)
        # Set the y-tick labels
        months = ["January", "February", "March", "April", "May", "June", "July", "August", "September", "October", "November", "December"]
        ax0.set_yticks(np.arange(len(months)) + 0.5)  # Centering the labels
        ax0.set_yticklabels(months)
        # Adjust the x-ticks and labels to start from 1
        xticks = np.arange(1, AG_sc.shape[1] + 1)
        ax0.set_xticks(xticks - 0.5, minor=False)  # Shift the ticks to be centered
        ax0.set_xticklabels(xticks)
        # Increase the size of y and x tick labels
        ax0.tick_params(axis='y', labelsize=22)
        ax0.tick_params(axis='x', labelsize=22)
        # Define colorbar axes and plot colorbar
        ax1 = plt.subplot(gs[1])
        cb = plt.colorbar(ax0.collections[0], cax=ax1, orientation='horizontal')
        # Increase the size of colorbar tick labels
        ax1.tick_params(labelsize=22)

    plt.show()
