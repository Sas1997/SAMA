import numpy as np

from EMS import energy_management
from Plot_Methods import *
from InputData import InData

def results(X):
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
    Tc = np.divide((InData.T + (InData.Tc_noct - InData.Ta_noct) * (InData.G / InData.G_noct) * (
            1 - (InData.n_PV * (1 - InData.Tcof * InData.Tref) / InData.gama))), (
                           1 + (InData.Tc_noct - InData.Ta_noct) * (InData.G / InData.G_noct) * (
                           (InData.Tcof * InData.n_PV) / InData.gama)))  # Module Temprature
    Ppv = np.multiply(InData.fpv * Pn_PV * (InData.G / InData.Gref),
                      (1 + InData.Tcof * (Tc - InData.Tref)))  # output power(kw) _hourly

    # %% Wind turbine Power Calculation
    v1 = InData.Vw  # hourly wind speed
    v2 = ((InData.h_hub / InData.h0) ** (
        InData.alfa_wind_turbine)) * v1  # v1 is the speed at a reference heightv2 is the speed at a hub height h2
    Pwt = np.zeros(8760)

    Pwt[v2 < InData.v_cut_in] = 0
    Pwt[v2 > InData.v_cut_out] = 0
    true_value = np.logical_and(InData.v_cut_in <= v2, v2 < InData.v_rated)
    Pwt[np.logical_and(InData.v_cut_in <= v2, v2 < InData.v_rated)] = v2[true_value] ** 3 * \
                                                                      (InData.Pwt_r / (
                                                                              InData.v_rated ** 3 - InData.v_cut_in ** 3)) - (
                                                                              InData.v_cut_in ** 3 / (
                                                                              InData.v_rated ** 3 - \
                                                                              InData.v_cut_in ** 3)) * (
                                                                          InData.Pwt_r)
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

    Eb = np.array(Eb)
    Pdg = np.array(Pdg)
    Edump = np.array(Edump)
    Ens = np.array(Ens)
    Pch = np.array(Pch)
    Pdch = np.array(Pdch)
    Pbuy = np.array(Pbuy)
    Psell = np.array(Psell)
    Pinv = np.array(Pinv)

    q = (InData.a * Pdg + InData.b * Pn_DG) * (Pdg > 0)  # Fuel consumption of a diesel generator

    # %% installation and operation cost

    # Total Investment cost ($)
    I_Cost = InData.C_PV * (1 - InData.RE_incentives) * Pn_PV + InData.C_WT * (1 - InData.RE_incentives) * Pn_WT + \
             InData.C_DG * Pn_DG + InData.C_B * (1 - InData.RE_incentives) * Cn_B + \
             InData.C_I * (1 - InData.RE_incentives) * Cn_I + InData.C_CH * (1 - InData.RE_incentives) + \
             InData.Engineering_Costs * (1 - InData.RE_incentives) * Pn_PV
    I_Cost_without_incentives = InData.C_PV * Pn_PV + InData.C_WT * Pn_WT + InData.C_DG * Pn_DG + InData.C_B * Cn_B + \
                                InData.C_I * Cn_I + InData.C_CH + InData.Engineering_Costs * Pn_PV
    Total_incentives_received = I_Cost_without_incentives - I_Cost

    Top_DG = np.sum(Pdg > 0) + 1
    L_DG = np.round(InData.TL_DG / Top_DG)
    RT_DG = np.ceil(InData.n / L_DG) - 1  # Replecement time

    # Total Replacement cost ($)
    R_Cost = 0
    for i in range(InData.n):
        # adding RC_PV to R_Cost
        if 0 <= (i + 1) % InData.L_PV < 1:
            R_Cost += InData.R_PV * Pn_PV / (1 + InData.ir) ** (1.001 * (i + 1))

        # adding RC_WT to R_Cost
        if 0 <= (i + 1) % InData.L_WT < 1:
            R_Cost += InData.R_WT * Pn_WT / (1 + InData.ir) ** (1.001 * (i + 1))

        # adding RC_DG to R_Cost
        if 0 <= (i + 1) % L_DG < 1:
            R_Cost += InData.R_DG * Pn_DG / (1 + InData.ir) ** (1.001 * (i + 1))

        # adding RC_B to R_Cost
        if 0 <= (i + 1) % InData.L_B < 1:
            R_Cost += InData.R_B * Cn_B / (1 + InData.ir) ** (1.001 * (i + 1))

        # adding RC_I to R_Cost
        if 0 <= (i + 1) % InData.L_I < 1:
            R_Cost += InData.R_I * Cn_I / (1 + InData.ir) ** (1.001 * (i + 1))

        # adding RC_CH to R_Cost
        if 0 <= (i + 1) % InData.L_CH < 1:
            R_Cost += InData.R_CH / (1 + InData.ir) ** (1.001 * (i + 1))

    # Total M&O Cost ($/year)
    MO_Cost = (InData.MO_PV * Pn_PV + InData.MO_WT * Pn_WT + InData.MO_DG * np.sum(Pn_DG > 0) + \
               InData.MO_B * Cn_B + InData.MO_I * Cn_I + InData.MO_CH) / (1 + InData.ir) ** np.array(
        range(1, InData.n + 1))

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
    DG_Emissions = np.sum(q * (InData.CO2 + InData.NOx + InData.SO2)) / 1000  # total emissions (kg/year)
    Grid_Emissions = np.sum(Pbuy * (InData.E_CO2 + InData.E_SO2 + InData.E_NOx)) / 1000  # total emissions (kg/year)

    Grid_Cost = (np.sum(Pbuy * InData.Cbuy) - np.sum(Psell * InData.Csell)) * 1 / (1 + InData.ir) ** np.array(
        range(1, InData.n + 1))

    # Capital recovery factor
    CRF = InData.ir * (1 + InData.ir) ** InData.n / ((1 + InData.ir) ** InData.n - 1)

    # Totall Cost
    NPC = I_Cost + np.sum(R_Cost) + np.sum(MO_Cost) + np.sum(C_Fu) - Salvage + np.sum(Grid_Cost) * (
            1 + InData.System_Tax)
    NPC_without_incentives = (I_Cost_without_incentives + R_Cost + sum(MO_Cost) + sum(C_Fu) - Salvage + sum(
        Grid_Cost)) * (1 + InData.System_Tax)

    Operating_Cost = CRF * (np.sum(R_Cost) + np.sum(MO_Cost) + np.sum(C_Fu) - Salvage + np.sum(Grid_Cost))

    LCOE = CRF * NPC / np.sum(InData.Eload - Ens + Psell)  # Levelized Cost of Energy ($/kWh)
    LCOE_without_incentives = CRF * NPC_without_incentives / sum(InData.Eload - Ens + Psell)
    LEM = (DG_Emissions + Grid_Emissions) / sum(InData.Eload - Ens)  # Levelized Emissions(kg/kWh)

    Ebmin = InData.SOC_min * Cn_B  # Battery minimum energy
    Pb_min = np.zeros(InData.NT)
    Pb_min[1: InData.NT] = (Eb[0:InData.NT - 1] - Ebmin) + Pdch[
                                                           1: InData.NT]  # Battery minimum power in t = 2, 3, ..., NT
    Ptot = (
                   Ppv + Pwt + Pb_min) * InData.n_I + Pdg + InData.Grid * InData.Pbuy_max  # total available power in system for each hour
    DE = np.maximum((InData.Eload - Ptot), 0)  # power shortage in each hour
    LPSP = np.sum(Ens) / np.sum(InData.Eload)

    RE = 1 - np.sum(Pdg + Pbuy) / np.sum(InData.Eload + Psell - Ens)
    if (np.isnan(RE)):
        RE = 0

    Investment = np.zeros(InData.n)
    Investment[0] = I_Cost
    Salvage1 = np.zeros(InData.n)
    Salvage1[InData.n - 1] = Salvage
    Salvage = Salvage1

    Operating = InData.MO_PV * Pn_PV + InData.MO_WT * Pn_WT + InData.MO_DG * Pn_DG + InData.MO_B * \
                Cn_B + InData.MO_I * Cn_I + sum(Pbuy * InData.Cbuy) - sum(Psell * InData.Csell)
    Operating = Operating * np.ones(InData.n)
    Fuel = sum(InData.C_fuel * q)

    RC_PV=np.zeros(InData.n)
    RC_WT = np.zeros(InData.n)
    RC_DG = np.zeros(InData.n)
    RC_B = np.zeros(InData.n)
    RC_I= np.zeros(InData.n)

    index = np.ceil(np.arange(InData.L_PV - 1, InData.n, InData.L_PV))
    index = np.array(index, dtype=int)
    RC_PV[index] = InData.R_PV * Pn_PV

    index = np.ceil(np.arange(InData.L_WT - 1, InData.n, InData.L_WT))
    index = np.array(index, dtype=int)
    RC_WT[index] = InData.R_WT * Pn_WT

    index = np.ceil(np.arange(L_DG - 1, InData.n, L_DG))
    index = np.array(index, dtype=int)
    RC_DG[index] = InData.R_DG * Pn_DG

    index = np.ceil(np.arange(InData.L_B - 1, InData.n, InData.L_B))
    index = np.array(index, dtype=int)
    RC_B[index] = InData.R_B * Cn_B

    index = np.ceil(np.arange(InData.L_I - 1, InData.n, InData.L_I))
    index = np.array(index, dtype=int)
    RC_I[index] = InData.R_I * Cn_I

    Replacement = RC_PV + RC_WT + RC_DG + RC_B + RC_I
    Cash_Flow = [-Investment, -Operating, Salvage, -Fuel * np.ones(InData.n), -Replacement]
    # ploting cash flow
    plot_cashflow(Cash_Flow)

    print(' ')
    print('System Size ')
    print(f'Cpv  (kW) = {Pn_PV}')
    if InData.WT == 1:
        print(f'Cwt  (kW) = {Pn_WT}')

    print(f'Cbat (kWh) = {Cn_B}')
    print(f'Cdg  (kW) = {Pn_DG}')
    print(f'Cinverter (kW) = {Cn_I}')

    print(' ')
    print('Result: ')
    print(f'NPC  = ${round(NPC, 2)}')
    print(f'NPC without incentives = ${round(NPC_without_incentives, 2)}')
    print(f'LCOE  = {round(LCOE, 2)}')
    print(f'LCOE without incentives = {round(LCOE_without_incentives, 2)}')
    print(f'Operation Cost  = ${round(Operating_Cost, 2)}')
    print(f'Initial Cost  = ${round(I_Cost, 2)}')
    print(f'Initial Cost without incentives= ${round(I_Cost_without_incentives, 2)}')
    print(f'Total incentives received= ${round(Total_incentives_received, 2)}')
    print(f'RE  = {100 * RE} % ')
    print(f'Total operation and maintenance cost  = ${sum(np.round(MO_Cost, 2))}')

    print(f'PV Power  = {sum(Ppv)} kWh')
    if InData.WT == 1:
        print(f'WT Power  = {sum(Pwt)} kWh')

    print(f'DG Power  = {sum(Pdg)} kWh')
    print(f'LPSP  = {100 * LPSP} %')
    print(f'Excess Electricity = {sum(Edump)} kWh')

    # New edits
    if InData.Grid == 1:
        Total_Pbuy = (sum(Pbuy) * InData.n) * (InData.Grid > 0)
        Total_Psell = (sum(Psell) * InData.n) * (InData.Grid > 0)
        print(f'Total power bought from Grid= {Total_Pbuy} kWh')
        print(f'Power sold to Grid= {Total_Psell} kWh')
        print(f'Total Money paid to the Grid= ${round(sum(Grid_Cost), 2)}')
        print(f'Grid Emissions   = {Grid_Emissions} (kg/year)')

    # ---
    print(f'Total Money paid by the user= ${np.round(np.sum(NPC), 2)}')
    # ---
    print(f'total fuel consumed by DG   = {sum(q)} (Liter/year)')
    print(f'DG Emissions   = {DG_Emissions} (kg/year)')
    print(f'LEM  = {LEM} kg/kWh')
    print("")

    # Plot Results
    if InData.Grid == 0:
        Pbuy = 0
        Psell = 0
    else:

        xlabel = 't(hour)'
        ylabel = 'Pgrid (kWh)'
        legends = ['Buy', 'Sell']
        vec = [Pbuy, Psell]
        title = ""
        plot_my_plots(vec, legends, title, xlabel, ylabel)

    # -------
    xlabel = 't(hour)'
    ylabel = ''
    legends = ['Load-Ens', 'Pdg', 'Pbat', 'P_{RE}']
    vec = [InData.Eload - Ens, Pdg, Pch - Pdch, Ppv + Pwt]
    title = ""
    plot_my_plots(vec, legends, title, xlabel, ylabel)

    # -------
    xlabel = 't(hour)'
    ylabel = 'SOC'
    legends = ['SOC']
    vec = [Eb / (Cn_B+0.0000001)]
    title = 'State of Charge'
    plot_my_plots(vec, legends, title, xlabel, ylabel)

    # Plot results for one specific day
    Day = 180
    t1 = Day * 24 + 1
    t2 = Day * 24 + 24

    n_row=4
    n_col=4
    title=f'Results for {Day}-th day'
    titles=[]
    indices=[]
    x_labels=[]
    y_labels=[]
    x_limits=[]
    values=[]
    is_bar=[]

    #subplot 1
    titles.append('Load Profile')
    y_labels.append('E_{load} [kWh]')
    x_labels.append('t[hour]')
    values.append(InData.Eload)
    indices.append((1,5))
    x_limits.append([t1,t2])
    is_bar.append(False)

    # subplot 2
    titles.append('Plane of Array Irradiance')
    y_labels.append('G[W/m^2]')
    x_labels.append('t[hour]')
    values.append(InData.T)
    indices.append(2)
    x_limits.append([t1, t2])
    is_bar.append(False)

    # subplot 3
    titles.append('Ambient Temperature')
    y_labels.append('T[^o C]')
    x_labels.append('t[hour]')
    values.append(InData.G)
    indices.append(6)
    x_limits.append([t1, t2])
    is_bar.append(False)

    # subplot 4
    titles.append('PV Power')
    y_labels.append('P_{pv} [kWh]')
    x_labels.append('t[hour]')
    values.append(Ppv)
    indices.append((3,8))
    x_limits.append([t1, t2])
    is_bar.append(False)

    if InData.WT == 1:
        # subplot 5
        titles.append('PV Power')
        y_labels.append('P_{pv} [kWh]')
        x_labels.append('t[hour]')
        values.append(Ppv)
        indices.append((3, 4))
        x_limits.append([t1, t2])
        is_bar.append(False)

        # subplot 6
        titles.append('WT Energy')
        y_labels.append('P_{wt} [kWh]')
        x_labels.append('t[hour]')
        values.append(Pwt)
        indices.append((7, 8))
        x_limits.append([t1, t2])
        is_bar.append(False)

    # subplot 7
    titles.append('Diesel Generator Energy')
    y_labels.append('E_{DG} [kWh]')
    x_labels.append('t[hour]')
    values.append(Pdg)
    indices.append((9, 10))
    x_limits.append([t1, t2])
    is_bar.append(False)

    # subplot 8
    titles.append('Battery Energy Level')
    y_labels.append('E_{b} [kWh]')
    x_labels.append('t[hour]')
    values.append(Eb)
    indices.append(11)
    x_limits.append([t1, t2])
    is_bar.append(False)

    # subplot 9
    titles.append('State of Charge')
    y_labels.append('SOC')
    x_labels.append('t[hour]')
    values.append(Eb / (Cn_B+0.000001))
    indices.append(12)
    x_limits.append([t1, t2])
    is_bar.append(False)

    # subplot 10
    titles.append('Loss of Power Suply')
    y_labels.append('LPS[kWh]')
    x_labels.append('t[hour]')
    values.append(Ens)
    indices.append(13)
    x_limits.append([t1, t2])
    is_bar.append(False)

    # subplot 11
    titles.append('Dumped Energy')
    y_labels.append('E_{dump} [kWh]')
    x_labels.append('t[hour]')
    values.append(Edump)
    indices.append(14)
    x_limits.append([t1, t2])
    is_bar.append(False)

    # subplot 12
    titles.append('Battery decharge Energy')
    y_labels.append('E_{dch} [kWh]')
    x_labels.append('t[hour]')
    values.append(Pdch)
    indices.append(15)
    x_limits.append([t1, t2])
    is_bar.append(True)

    # subplot 13
    titles.append('Battery charge Energy')
    y_labels.append('E_{ch} [kWh]')
    x_labels.append('t[hour]')
    values.append(Pch)
    indices.append(16)
    x_limits.append([t1, t2])
    is_bar.append(True)

    plot_one_day(title, n_row, n_col, titles, indices, x_labels, y_labels, x_limits, values,is_bar)

    C_c = np.zeros((12, 31))
    index = 0
    for m in range(12):
        index1 = index
        for d in range(InData.daysInMonth[m]):
            Total_daily_load = sum(InData.Eload[index1:index1 + 24])
            C_c[m, d] = Total_daily_load
            index1 += 24
        index = (24 * InData.daysInMonth[m]) + index

    EE_c = np.round(LCOE * C_c, 2)

    # figure 6
    YData = ["January","February","March","April","May","June","July","August","September","October","November","December"]
    EE_c[EE_c == 0] = None
    plot_heatmap(title, EE_c,YData)


    # Saving
    paperout = [Ppv, Pdg, Pch, Pdch]
    np.savetxt('Output.csv', paperout, delimiter=',')
    return Psell


