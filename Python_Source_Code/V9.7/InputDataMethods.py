from math import ceil
import numpy as np


# calcSeasonalRate
def calcSeasonalRate(prices, months, daysInMonth):
    Cbuy = np.zeros(8760)
    hCount = 1

    for m in range(12):
        hoursStart = hCount
        hoursEnd = hoursStart + (24 * daysInMonth[m])
        hoursRange = np.array(range(hoursStart, hoursEnd - 1))
        Cbuy[hoursRange] = prices[int(months[m])]
        hCount = hoursEnd
    return Cbuy


# calcFlatRate
def calcFlatRate(price):
    Cbuy = np.zeros(8760)
    for h in range(8760):
        Cbuy[h] = price
    return Cbuy


# calcMonthlyRate
def calcMonthlyRate(prices, daysInMonth):
    Cbuy = np.zeros(8760)
    hCount = 0
    for m in range(12):
        for h in range(24 * daysInMonth[m]):
            Cbuy[hCount] = prices[m]
            hCount += 1
    return Cbuy


# calcTieredRate
def calcTieredRate(prices, tierMax, load, daysInMonth):
    Cbuy = np.zeros(8760)
    hCount = 0
    for m in range(12):
        monthlyLoad = 0
        for h in range(24 * daysInMonth[m]):
            monthlyLoad += load[hCount]
            if monthlyLoad < tierMax[0]:
                Cbuy[hCount] = prices[0]
            elif monthlyLoad < tierMax[1]:
                Cbuy[hCount] = prices[1]
            else:
                Cbuy[hCount] = prices[2]
            hCount += 1
    return Cbuy


# calcSeasonalTieredRate
def calcSeasonalTieredRate(prices, tierMax, load, months, daysInMonth):
    Cbuy = np.zeros(8760)
    hCount = 0
    for m in range(12):
        monthlyLoad = 0
        months_m = int(months[m])
        for h in range(24 * daysInMonth[m]):
            monthlyLoad += load[hCount]
            if monthlyLoad < tierMax[months_m, 0]:
                Cbuy[hCount] = prices[months_m, 0]
            elif monthlyLoad < tierMax[months_m, 1]:
                Cbuy[hCount] = prices[months_m, 1]
            else:
                Cbuy[hCount] = prices[months_m, 2]
            hCount += 1
    return Cbuy


# calcMonthlyTieredRate
def calcMonthlyTieredRate(prices, tierMax, load, daysInMonth):
    Cbuy = np.zeros(8760)
    hCount = 0
    for m in range(12):
        monthlyLoad = 0
        for h in range(24 * daysInMonth[m]):
            monthlyLoad += load[hCount]
            if monthlyLoad < tierMax[m, 0]:
                Cbuy[hCount] = prices[m, 0]
            elif monthlyLoad < tierMax[m, 1]:
                Cbuy[hCount] = prices[m, 1]
            else:
                Cbuy[hCount] = prices[m, 2]
            hCount += 1
    return Cbuy


# calcTouRate
def calcTouRate(onPrice, midPrice, offPrice, onHours, midHours, offHours, Month, Day, holidays):
    Cbuy = np.zeros(8760)
    for m in range(12):
        if m == 0:
            t_start = 0
            t_end = 24 * Day[m]
        else:
            t_start = 24 * Day[m - 1]
            t_end = 24 * Day[m]

        if Month[m] == 0:  # for summer

            tp = onHours[0, :]
            tm = midHours[0, :]
            toff = offHours[0, :]
            P_peak = onPrice[0]
            P_mid = midPrice[0]
            P_offpeak = offPrice[0]

        else:  # for winter
            tp = onHours[1, :]
            tm = midHours[1, :]
            toff = offHours[1, :]
            P_peak = onPrice[1]
            P_mid = midPrice[1]
            P_offpeak = offPrice[1]

        for j in range(t_start, t_end):
            h = t_end - t_start + j
            if h in tp:
                Cbuy[j] = P_peak  # set onhours to P_peak
            elif h in tm:
                Cbuy[j] = P_mid  # set midHours to P_mid
            else:
                Cbuy[j] = P_offpeak  # set all hours to offpeak by default

    for d in range(365):
        if d % 7 >= 5:
            st = 24 * d + 1
            ed = 24 * (d + 1)
            Cbuy[st: ed] = P_offpeak

    holidays = holidays - 1
    for d in range(365):
        if d in holidays:
            st = 24 * d + 1
            ed = 24 * (d + 1)
            Cbuy[st: ed] = P_offpeak
    return Cbuy


# other_parameters (Input_Data of matlab code)
def other_parameters(Eload, Eload_Previous):
    # Type of system (1: included, 0=not included)
    PV = 1
    WT = 1
    DG = 1
    Bat = 1
    Grid = 0

    EM = 0  # 0: LCOE, 1:LCOE+LEM

    Budget = 200e3  # Limit On Total Capital Cost
    # Economic Parameters
    n = 25  # Life year of system (year)

    n_ir_rate = 4.5  # Nominal discount rate
    n_ir = n_ir_rate / 100
    e_ir_rate = 2  # Expected inflation rate
    e_ir = e_ir_rate / 100

    ir = (n_ir - e_ir) / (1 + e_ir)  # real discount rate

    Tax_rate = 0  # Equipment sale tax Percentage
    System_Tax = Tax_rate / 100

    RE_incentives_rate = 30  # Federal tax credit percentage
    RE_incentives = RE_incentives_rate / 100
    # Constraints
    LPSP_max_rate = 0  # Maximum loss of power supply probability
    LPSP_max = LPSP_max_rate / 100

    RE_min_rate = 75  # Minimum Renewable Energy Capacity
    RE_min = RE_min_rate / 100
    # Technical data
    # Rated capacity
    Ppv_r = 0.5  # PV module rated power (kW)
    Pwt_r = 1  # WT rated power (kW)
    Cbt_r = 1  # Battery rated Capacity (kWh)
    Cdg_r = 0.5  # DG rated Capacity (kW)
    # PV
    # hourly_solar_radiation W
    fpv = 0.9  # the PV derating factor [#]
    Tcof = 0  # temperature coefficient
    Tref = 25  # temperature at standard test condition
    Tc_noct = 46.5  # Nominal operating cell temperature
    Ta_noct = 20
    G_noct = 800
    gama = 0.9
    n_PV = 0.205  # Efficiency of PV module
    Gref = 1000  # 1000 W/m^2
    # D_PV=0.01        # PV yearly degradation
    L_PV = 25  # Life time (year)
    RT_PV = ceil(n / L_PV) - 1  # Replacement time
    # Inverter
    n_I = 0.96  # Efficiency
    L_I = 25  # Life time (year)
    RT_I = ceil(n / L_I) - 1  # Replacement time
    # WT data
    h_hub = 17  # Hub height
    h0 = 43.6  # anemometer height
    nw = 1  # Electrical Efficiency
    v_cut_out = 25  # cut out speed
    v_cut_in = 2.5  # cut in speed
    v_rated = 9.5  # rated speed(m/s)
    alfa_wind_turbine = 0.14  # coefficient of friction ( 0.11 for extreme wind conditions, and 0.20 for normal wind conditions)
    # n_WT=0.30        # Efficiency of WT module
    # D_WT=0.05        # WT yearly degradation
    L_WT = 20  # Life time (year)
    RT_WT = ceil(n / L_WT) - 1  # Replacement time
    # Diesel generator
    # n_DG=0.4         # Efficiency
    # D_DG=0.05        # yearly degradation (#)
    LR_DG = 0.25  # Minimum Load Ratio (#)
    # Diesel Generator fuel curve
    a = 0.2730  # L/hr/kW output
    b = 0.0330  # L/hr/kW rated
    TL_DG = 24000  # Life time (h)
    # Emissions produced by Diesel generator for each fuel in litre [L]	g/L
    CO2 = 2621.7
    CO = 16.34
    NOx = 6.6
    SO2 = 20
    # Battery
    SOC_min = 0.2
    SOC_max = 1
    SOC_initial = 0.5
    # D_B=0.05               # Degradation
    Q_lifetime = 8000  # kWh
    self_discharge_rate = 0  # Hourly self-discharge rate
    alfa_battery = 1  # is the storage's maximum charge rate [A/Ah]
    c = 0.403  # the storage capacity ratio [unless]
    k = 0.827  # the storage rate constant [h-1]
    Imax = 16.7  # the storage's maximum charge current [A]
    Vnom = 12  # the storage's nominal voltage [V]
    ef_bat = 0.8  # Round trip efficiency
    L_B = 7.5  # Life time (year)
    RT_B = ceil(n / L_B) - 1  # Replacement time
    # Charger
    L_CH = 25  # Life time (year)
    RT_CH = ceil(n / L_CH) - 1  # Replacement time

    # Pricing method


    Pricing_method = 1  # 1=Top down 2=bottom up

    # Top-down price definition
    if Pricing_method == 1:
        # Pricing method 1/top down
        Total_PV_price = 2950
        # NREL percentages
        r_PV = 0.1812
        r_inverter = 0.1492
        r_Installation_cost = 0.0542
        r_Overhead = 0.0881
        r_Sales_and_marketing = 0.1356
        r_Permiting_and_Inspection = 0.0712
        r_Electrical_BoS = 0.1254
        r_Structrual_BoS = 0.0542
        r_Profit_costs = 0.1152
        r_Sales_tax = 0.0271
        r_Supply_Chain_costs = 0
        # Engineering Costs (Per/kW)
        Installation_cost = Total_PV_price * r_Installation_cost
        Overhead = Total_PV_price * r_Overhead
        Sales_and_marketing = Total_PV_price * r_Sales_and_marketing
        Permiting_and_Inspection = Total_PV_price * r_Permiting_and_Inspection
        Electrical_BoS = Total_PV_price * r_Electrical_BoS
        Structrual_BoS = Total_PV_price * r_Structrual_BoS
        Profit_costs = Total_PV_price * r_Profit_costs
        Sales_tax = Total_PV_price * r_Sales_tax
        Supply_Chain_costs = Total_PV_price * r_Supply_Chain_costs
        Engineering_Costs = Sales_tax + Profit_costs + Installation_cost + Overhead + Sales_and_marketing + Permiting_and_Inspection + Electrical_BoS + Structrual_BoS + Supply_Chain_costs
        # PV
        C_PV = Total_PV_price * r_PV  # Capital cost ($) per KW
        R_PV = Total_PV_price * r_PV  # Replacement Cost of PV modules Per KW
        MO_PV = 28.12  # PV O&M  cost ($/year/kw)
        # Inverter
        C_I = Total_PV_price * r_inverter  # Capital cost ($/kW)
        R_I = Total_PV_price * r_inverter  # Replacement cost ($/kW)
        MO_I = 3  # Inverter O&M cost ($/kW.year)
        # WT
        C_WT = 1200  # Capital cost ($) per KW
        R_WT = 1200  # Replacement Cost of WT Per KW
        MO_WT = 40  # O&M  cost ($/year/kw)
        # Diesel generator
        C_DG = 240.45  # Capital cost ($/KW)
        R_DG = 240.45  # Replacement Cost ($/kW)
        MO_DG = 0.064  # O&M+ running cost ($/op.h)
        C_fuel = 1.39  # Fuel Cost ($/L)
        # Battery
        C_B = 458.06  # Capital cost ($/kWh)
        R_B = 458.06  # Replacement Cost ($/kW)
        MO_B = 10  # Maintenance cost ($/kWh.year)
        # Charger
        if Bat == 1:
            C_CH = 149.99  # Capital Cost ($)
            R_CH = 149.99  # Replacement Cost ($)
            MO_CH = 0  # O&M cost ($/year)
        else:
            C_CH = 0  # Capital Cost ($)
            R_CH = 0  # Replacement Cost ($)
            MO_CH = 0  # O&M cost ($/year)

    else:  # Pricing method 2/bottom up
        # Engineering Costs (Per/kW)
        Installation_cost = 160
        Overhead = 260
        Sales_and_marketing = 400
        Permiting_and_Inspection = 210
        Electrical_BoS = 370
        Structrual_BoS = 160
        Supply_Chain_costs = 0
        Profit_costs = 340
        Sales_tax = 80
        Engineering_Costs = Sales_tax + Profit_costs + Installation_cost + Overhead + Sales_and_marketing + Permiting_and_Inspection + Electrical_BoS + Structrual_BoS + Supply_Chain_costs
        # PV
        C_PV = 540  # Capital cost ($) per KW
        R_PV = 540  # Replacement Cost of PV modules Per KW
        MO_PV = 29.49  # O&M  cost ($/year/kw)
        # Inverter
        C_I = 440  # Capital cost ($/kW)
        R_I = 440  # Replacement cost ($/kW)
        MO_I = 0  # O&M cost ($/kw.year)
        # WT
        C_WT = 1200  # Capital cost ($) per KW
        R_WT = 1200  # Replacement Cost of WT Per KW
        MO_WT = 40  # O&M  cost ($/year/kw)
        # Diesel generator
        C_DG = 240.45  # Capital cost ($/KW)
        R_DG = 240.45  # Replacement Cost ($/kW)
        MO_DG = 0.064  # O&M+ running cost ($/op.h)
        C_fuel = 1.39  # Fuel Cost ($/L)
        # Battery
        C_B = 458.06  # Capital cost ($/KWh)
        R_B = 458.06  # Repalacement Cost ($/kW)
        MO_B = 0  # Maintenance cost ($/kw.year)
        # Charger
        if Bat == 1:
            C_CH = 149.99  # Capital Cost ($)
            R_CH = 149.99  # Replacement Cost ($)
            MO_CH = 0  # O&M cost ($/year)
        else:
            C_CH = 0  # Capital Cost ($)
            R_CH = 0  # Replacement Cost ($)
            MO_CH = 0  # O&M cost ($/year)

    # Prices for Utility
    #
    # Months
    months = np.ones(12)

    # days in each month
    daysInMonth = np.array([31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31])
    # Definition
    # 1 = flat rate
    # 2 = seasonal rate
    # 3 = monthly rate
    # 4 = tiered rate
    # 5 = seasonal tiered rate
    # 6 = monthly tiered rate
    # 7 = time of use rate
    # Grid emission information
    # Emissions produced by Grid generators (g/kW)
    E_CO2 = 1.43
    E_SO2 = 0.01
    E_NOx = 0.39
    # Define hourly rate structure
    rateStructure = 6
    # Fixed expenses
    Annual_expenses = 0
    Grid_sale_tax_rate = 9.5
    Grid_Tax = Grid_sale_tax_rate / 100
    # Monthly fixed charge
    Monthly_fixed_charge_system = 2  # 1:flat 2:tier based
    # Service Charge (SC)
    if Monthly_fixed_charge_system == 1:
        # Flat
        SC_flat = 23.5
        Service_charge = SC_flat * np.ones(12)
    else:
        # Tiered
        totalmonthlyload = np.zeros(12)
        SC_1 = 2.30  # tier 1 service charge
        Limit_SC_1 = 500  # limit for tier 1
        SC_2 = 7.9  # tier 2 service charge
        Limit_SC_2 = 1500  # limit for tier 2
        SC_3 = 22.7  # tier 3 service charge
        Limit_SC_3 = 1500  # limit for tier 3
        SC_4 = 22.7  # tier 4 service charge
        hourCount = 0
        for m in range(12):
            monthlyLoad = 0
            for h in range(24 * daysInMonth[m]):
                monthlyLoad += Eload_Previous[hourCount]
                hourCount += 1
            totalmonthlyload[m] = monthlyLoad

        if np.max(totalmonthlyload) < Limit_SC_1:
            Service_charge = SC_1 * np.ones(12)
        elif np.max(totalmonthlyload) < Limit_SC_2:
            Service_charge = SC_2 * np.ones(12)
        elif np.max(totalmonthlyload) < Limit_SC_3:
            Service_charge = SC_3 * np.ones(12)
        else:
            Service_charge = SC_4 * np.ones(12)

    # Hourly charges
    if rateStructure == 1:  # flat rate
        # price for flat rate
        flatPrice = 0.112
        Cbuy = flatPrice * np.ones(8760)

    elif rateStructure == 2:  # seasonal rate
        # prices for seasonal rate [summer, winter]
        seasonalPrices = np.array([0.17, 0.13])
        # define summer season
        months[4:10] = 0
        months[0:3] = 1
        months[11] = 1
        Cbuy = calcSeasonalRate(seasonalPrices, months, daysInMonth)

    elif rateStructure == 3:  # monthly rate
        # prices for monthly rate [Jan-Dec]
        monthlyPrices = np.array([0.15, 0.14, 0.13, 0.16, 0.11, 0.10, 0.12, 0.13, 0.14, 0.10, 0.15, 0.16])
        Cbuy = calcMonthlyRate(monthlyPrices, daysInMonth)

    elif rateStructure == 4:  # tiered rate
        # prices and max kwh limits [tier 1, 2, 3]
        tieredPrices = np.array([0.1, 0.12, 0.15])
        tierMax = np.array([680, 720, 1050])
        Cbuy = calcTieredRate(tieredPrices, tierMax, Eload, daysInMonth)

    elif rateStructure == 5:  # seasonal tiered rate
        # prices and max kwh limits [summer,winter][tier 1, 2, 3]
        seasonalTieredPrices = np.array([[0.05, 0.08, 0.14],
                                         [0.09, 0.13, 0.2]])
        seasonalTierMax = np.array([[400, 800, 4000],
                                    [1000, 1500, 4000]])
        # define summer season
        months[4:10] = 0
        months[0:3] = 1
        months[11] = 1
        Cbuy = calcSeasonalTieredRate(seasonalTieredPrices, seasonalTierMax, Eload, months, daysInMonth)

    elif rateStructure == 6:  # monthly tiered rate
        # prices and max kwh limits [Jan-Dec][tier 1, 2, 3]
        monthlyTieredPrices = np.array([
            [0.19488, 0.25347, 0.25347],
            [0.19488, 0.25347, 0.25347],
            [0.19488, 0.25347, 0.25347],
            [0.19375, 0.25234, 0.25234],
            [0.19375, 0.25234, 0.25234],
            [0.19375, 0.25234, 0.33935],
            [0.18179, 0.24038, 0.32739],
            [0.18179, 0.24038, 0.32739],
            [0.18179, 0.24038, 0.32739],
            [0.19192, 0.25051, 0.25051],
            [0.19192, 0.25051, 0.25051],
            [0.19192, 0.25051, 0.25051]])

        monthlyTierLimits = np.array([
            [500, 1500, 1501],
            [500, 1500, 1501],
            [500, 1500, 1501],
            [500, 1500, 1501],
            [500, 1500, 1501],
            [500, 1500, 1501],
            [500, 1500, 1501],
            [500, 1500, 1501],
            [500, 1500, 1501],
            [500, 1500, 1501],
            [500, 1500, 1501],
            [500, 1500, 1501]])
        Cbuy = calcMonthlyTieredRate(monthlyTieredPrices, monthlyTierLimits, Eload, daysInMonth)

    elif rateStructure == 7:  # time of use rate

        # prices and time of use hours [summer,winter]
        onPrice = np.array([0.17, 0.17])
        midPrice = np.array([0.113, 0.113])
        offPrice = np.array([0.083, 0.083])
        onHours = np.array([[11, 12, 13, 14, 15, 16], [7, 8, 9, 10, 17, 18]])
        midHours = np.array([[7, 8, 9, 10, 17, 18], [11, 12, 13, 14, 15, 16]])
        offHours = np.array([[1, 2, 3, 4, 5, 6, 19, 20, 21, 22, 23, 24], [1, 2, 3, 4, 5, 6, 19, 20, 21, 22, 23, 24]])
        # define summer season
        months[4:7] = 1
        # Holidays definition based on the number of the day in 365 days format
        holidays = np.array([10, 50, 76, 167, 298, 340])
        Cbuy = calcTouRate(onPrice, midPrice, offPrice, onHours, midHours, offHours, months, daysInMonth, holidays)

    # Sell electricity to the grid
    Csell = 0.14
    # Constraints for selling to grid
    Pbuy_max = ceil(1.2 * max(Eload))  # kWh
    Psell_max = Pbuy_max


    # returning all required variables
    return Ppv_r,Pwt_r,Cbt_r,Cdg_r,Tc_noct,fpv,Gref,Tcof,Tref,h_hub,h0,alfa_wind_turbine,v_cut_in,v_cut_out,v_rated,\
           R_B,Q_lifetime,ef_bat,b,C_fuel,R_DG,TL_DG,MO_DG,SOC_max,SOC_min,SOC_initial,n_I,Grid,Cbuy,a,LR_DG,Pbuy_max,\
           Psell_max,self_discharge_rate,alfa_battery,c,k,Imax,Vnom,C_PV,C_WT,C_DG,C_B,C_I,C_CH,n,R_PV,ir,L_PV,R_WT,\
           L_WT,L_B,R_I,R_CH,MO_PV,MO_WT,MO_B,MO_I,MO_CH,RT_PV,RT_WT,RT_B,RT_I,L_I,RT_CH,L_CH,CO2,NOx,SO2,E_CO2,E_SO2,\
           E_NOx,Csell,EM,LPSP_max,RE_min,Budget,Ta_noct,G_noct,n_PV,gama,PV, WT, Bat,DG,RE_incentives,\
           Engineering_Costs,System_Tax,Grid_Tax,daysInMonth


