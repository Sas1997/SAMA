import numpy as np
import pandas as pd
from math import ceil
from daysInMonth import daysInMonth

# Optimization
# PSO Parameters
class Input_Data:
    def __init__(self):
        self.MaxIt = 200  # Maximum Number of Iterations
        self.nPop = 50  # Population Size (Swarm Size)
        self.w = 1  # Inertia Weight
        self.wdamp = 0.99  # Inertia Weight Damping Ratio
        self.c1 = 2 # Personal Learning Coefficient
        self.c2 = 2  # Global Learning Coefficient

        # Multi-run
        self.Run_Time = 1  # Total number of runs in simulation

        # Calendar
        self.n = 25  # Lifetime of system in simulations (years)
        self.year = 2023  # Specify the desired year
        # Days in each month
        self.daysInMonth = daysInMonth(self.year)

        # Reading global inputs

        # Electrical load definitions
        # 1=Hourly load based on the CSV file
        # 2=Monthly hourly average load
        # 3=Monthly daily average load
        # 4=Monthly total load
        # 5=Scaled generic load based on Monthly total load
        # 6=Annual hourly average load
        # 7=Annual daily average load
        # 8=Scaled generic load based on annual total load
        # 9=Exactly equal to generic load

        load_type = 1  # Determine the way you want to input the electrical load by choosing one of the numbers above

        if load_type == 1:
            self.path_Eload = 'content/Eload.csv'
            self.EloadData = pd.read_csv(self.path_Eload, header=None).values
            self.Eload = np.array(self.EloadData[:, 0])

        elif load_type == 2:
            self.Monthly_haverage_load = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12])  # Define the monthly hourly averages for load here

            from dataextender import dataextender
            self.Eload = dataextender(self.daysInMonth, self.Monthly_haverage_load)

        elif load_type == 3:
            self.Monthly_daverage_load = np.array([10, 20, 31, 14, 15, 16, 17, 18, 19, 10, 11, 12])  # Define the monthly daily averages for load here

            self.Monthly_haverage_load = self.Monthly_daverage_load / 24
            from dataextender import dataextender
            self.Eload = dataextender(self.daysInMonth, self.Monthly_haverage_load)

        elif load_type == 4:
            self.Monthly_total_load = np.array([321, 223, 343, 423, 544, 623, 237, 843, 239, 140, 121, 312])  # Define the monthly total load here

            self.Monthly_haverage_load = self.Monthly_total_load / (self.daysInMonth * 24)
            from dataextender import dataextender
            self.Eload = dataextender(self.daysInMonth, self.Monthly_haverage_load)

        elif load_type == 5: # Based on the generic load
            peak_month = 'July'
            user_defined_load = np.array([300, 350, 320, 320, 320, 320, 320, 320, 320, 320, 320, 320]) # Define the monthly total load here

            from generic_load import generic_load
            self.Eload = generic_load(load_type, 1, peak_month, self.daysInMonth, user_defined_load)

        elif load_type == 6:
            self.Annual_haverage_load = 1  # Define the annual hourly average for load here

            self.Eload = np.full(8760, self.Annual_haverage_load)

        elif load_type == 7:
            self.Annual_daverage_load = 10  # Define the annual hourly average for load here

            self.Annual_haverage_load = self.Annual_daverage_load / 24
            self.Eload = np.full(8760, self.Annual_haverage_load)

        elif load_type == 8: #Annual total load
            self.Annual_total_load = 25660.72  # Define the annual hourly average for load here
            peak_month = 'July'

            from generic_load import generic_load
            self.Eload = generic_load(load_type, 1, peak_month, self.daysInMonth, self.Annual_total_load)

        else:
            peak_month = 'July'

            from generic_load import generic_load
            self.Eload = generic_load(load_type, 1, peak_month, self.daysInMonth, 1)

        # Previous year Electrical load definitions

        # 1=Hourly load equals to the current year
        # 2=Hourly load based on the CSV file
        # 3=Monthly hourly average load
        # 4=Monthly daily average load
        # 5=Monthly total load
        # 6=Scaled generic load based on Monthly total load
        # 7=Annual hourly average load
        # 8=Annual daily average load
        # 9=Scaled generic load based on annual total load
        # 10=Exactly equals to generic load

        load_previous_year_type = 1 # Determine the way you want to input the electrical load for previous year by choosing one of the numbers above

        if load_previous_year_type == 1:

            self.Eload_Previous = self.Eload

        elif load_previous_year_type == 2:

            self.path_Eload_Previous = 'content/Eload_previousyear.csv'
            self.Eload_PreviousData = pd.read_csv(self.path_Eload_Previous, header=None).values
            self.Eload_Previous = np.array(self.EloadData[:, 0])

        elif load_previous_year_type == 3:
            self.Monthly_haverage_load_previous = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12])  # Define the monthly hourly averages for load here

            from dataextender import dataextender
            self.Eload_Previous = dataextender(self.daysInMonth, self.Monthly_haverage_load_previous)

        elif load_previous_year_type == 4:
            self.Monthly_daverage_load_previous = np.array([10, 20, 31, 14, 15, 16, 17, 18, 19, 10, 11, 12])  # Define the monthly hourly averages for load here
            self.Monthly_haverage_load_previous = self.Monthly_daverage_load_previous / 24

            from dataextender import dataextender
            self.Eload_Previous = dataextender(self.daysInMonth, self.Monthly_haverage_load_previous)

        elif load_previous_year_type == 5:
            self.Monthly_total_load_previous = np.array([321, 223, 343, 423, 544, 623, 237, 843, 239, 140, 121, 312])  # Define the monthly total load here

            self.Monthly_haverage_load_previous = self.Monthly_total_load_previous / (self.daysInMonth * 24)
            from dataextender import dataextender
            self.Eload_Previous = dataextender(self.daysInMonth, self.Monthly_haverage_load_previous)

        elif load_previous_year_type == 6:
            peak_month = 'July'
            user_defined_load = np.array([300, 350, 320, 320, 320, 320, 320, 320, 320, 320, 320, 320])  # Define the monthly total load here

            from generic_load import generic_load
            self.Eload_Previous = generic_load(1, load_previous_year_type, peak_month, self.daysInMonth, user_defined_load)

        elif load_previous_year_type == 7:
            self.Annual_haverage_load_previous = 1  # Define the annual hourly average for load here

            self.Eload_Previous = np.full(8760, self.Annual_haverage_load_previous)

        elif load_previous_year_type == 8:
            self.Annual_daverage_load_previous = 10  # Define the annual hourly average for load here

            self.Annual_haverage_load_previous = self.Annual_daverage_load_previous / 24
            self.Eload_Previous = np.full(8760, self.Annual_haverage_load_previous)

        elif load_previous_year_type == 9: # Annual total load for previous year
            self.Annual_total_load = 12000  # Define the annual hourly average for load here
            peak_month = 'July'

            from generic_load import generic_load
            self.Eload_Previous = generic_load(1, load_previous_year_type, peak_month, self.daysInMonth, self.Annual_total_load)

        else:
            peak_month = 'July'

            from generic_load import generic_load
            self.Eload_Previous = generic_load(1, load_previous_year_type, peak_month, self.daysInMonth, 1)

        # Irradiance definitions
        # 1=Hourly irradiance based on POA calculator
        # 2=Hourly POA irradiance based on the user CSV file
        weather_url = 'content/METEO.csv'
        azimuth = 180
        tilt = 34  # Tilt angle of PV modules
        soiling = 5  # Soiling losses in percentage

        G_type = 1 # Determine the way you want to input the Irradiance by choosing one of the numbers above

        if G_type == 1:

            from sam_monofacial_poa import runSimulation
            temp_result = runSimulation(weather_url, tilt, azimuth, soiling)
            G_pd_to_numpy = temp_result[0]
            self.G = G_pd_to_numpy.values

        elif G_type == 2: # It should be Plane of array irradiance

            self.path_G = 'content/Irradiance.csv'
            self.GData = pd.read_csv(self.path_G, header=None).values
            self.G = np.array(self.GData[:, 0])

        # Temperature definitions
        # 1=Hourly Temperature based on the NSEDB file
        # 2=Hourly Temperature based on the user CSV file
        # 3=Monthly average Temperature
        # 4=Annual average Temperature

        T_type = 1 # Determine the way you want to input the Temperature by choosing one of the numbers above

        if T_type == 1:

            from sam_monofacial_poa import runSimulation
            temp_result = runSimulation(weather_url, tilt, azimuth, soiling)
            T_pd_to_numpy = temp_result[1]
            self.T = T_pd_to_numpy.values

        elif T_type == 2:

            self.path_T = 'content/Temperature.csv'
            self.TData = pd.read_csv(self.path_T, header=None).values
            self.T = np.array(self.TData[:, 0])

        elif T_type == 3:

            self.Monthly_average_temperature = np.array([-2, -5, -2, 1, 3, 6, 15, 22, 27, 23, 16, 7])  # Define the monthly hourly averages for temperature here

            from dataextender import dataextender
            self.T = dataextender(self.daysInMonth, self.Monthly_average_temperature)

        else: # Annual average Temperature
            self.Annual_average_temperature = 12

            self.T = np.full(8760, self.Annual_average_temperature)

        # Wind speed definitions
        # 1=Hourly Wind speed based on the NSEDB file
        # 2=Hourly Wind speed based on the user CSV file
        # 3=Monthly average Wind speed
        # 4=Annual average Wind speed

        WS_type = 1 # Determine the way you want to input the Wind speed by choosing one of the numbers above

        if WS_type == 1:

            from sam_monofacial_poa import runSimulation
            temp_result = runSimulation(weather_url, tilt, azimuth, soiling)
            WS_pd_to_numpy = temp_result[2]
            self.Vw = WS_pd_to_numpy.values

        elif WS_type == 2:

            self.path_WS = 'content/WSPEED.csv'
            self.WSData = pd.read_csv(self.path_WS, header=None).values
            self.Vw = np.array(self.WSData[:, 0])

        elif WS_type == 3:
            self.Monthly_average_windspeed = np.array([14.1, 21, 12.2, 31, 12.2, 11.2, 12.1, 13, 21, 9.2, 12.3, 18.1])  # Define the monthly hourly averages for load here

            from dataextender import dataextender
            self.Vw = np.array(dataextender(self.daysInMonth, self.Monthly_average_windspeed))

        else: # Annual average Wind speed

            self.Annual_average_windspeed = 10
            self.Vw = np.full(8760, self.Annual_average_windspeed)

        data = {'Eload': self.Eload, 'G': self.G, 'T': self.T, 'Vw': self.Vw}
        df = pd.DataFrame(data)
        df.to_csv('output/data/Inputs.csv', index=False)

        # Other inputs
        # Technical data
        # Type of system (1: included, 0=not included)
        self.PV = 1
        self.WT = 0
        self.DG = 0
        self.Bat = 1
        self.Lead_acid = 0
        self.Li_ion = 1
        self.Grid = 1
        # Net metering scheme
        # If compensation is towrds credits in Net metering, not monetary comenstation, by choosing the below option (putting NEM equals to 1), the yearly credits will be reconciled after 12 months
        self.NEM = 1

        if self.Grid == 0:
            self.NEM = 0

        # Constraints
        self.LPSP_max_rate = 0.0999999  # Maximum loss of power supply probability percentage

        self.LPSP_max = self.LPSP_max_rate / 100

        self.RE_min_rate = 75  # Minimum Renewable Energy Capacity percentage

        self.RE_min = self.RE_min_rate / 100

        self.EM = 0  # 0: NPC, 1:NPC+LEM

        # PV
        self.fpv = 0.9       # the PV derating factor [%]
        self.Tcof = -0.3        # temperature coefficient [%/C]
        self.Tref = 25       # temperature at standard test condition
        self.Tc_noct = 45    # Nominal operating cell temperature
        self.Ta_noct = 20
        self.G_noct = 800
        self.gama = 0.9
        self.n_PV = 0.2182   # Efficiency of PV module
        self.Gref = 1000     # 1000 W/m^2
        self.L_PV = 25       # Life time (year)
        self.RT_PV = ceil(self.n/self.L_PV) - 1   # Replacement time

        # Inverter
        self.n_I = 1     # Efficiency
        self.L_I = 25        # Life time (year)
        self.DC_AC_ratio = 1.99     # Maximum acceptable DC to AC ratio
        self.RT_I = ceil(self.n/self.L_I) - 1    # Replacement time

        # WT data
        self.h_hub = 17              # Hub height
        self.h0 = 43.6               # anemometer height
        self.nw = 1                  # Electrical Efficiency
        self.v_cut_out = 25          # cut out speed
        self.v_cut_in = 2.5          # cut in speed
        self.v_rated = 9.5           # rated speed(m/s)
        self.alfa_wind_turbine = 0.14       # Coefficient of friction (0.11 for extreme wind conditions, and 0.20 for normal wind conditions)
        self.L_WT = 20           # Life time (year)
        self.RT_WT = ceil(self.n/self.L_WT) - 1    # Replecement time

        # Diesel generator
        self.LR_DG = 0.25        # Minimum Load Ratio (%)

        # Diesel Generator fuel curve
        self.a = 0.2730          # L/hr/kW output
        self.b = 0.0330          # L/hr/kW rated

        self.TL_DG = 24000       # Life time (h)

        # Emissions produced by Diesel generator for each fuel in litre [L] g/L
        self.CO2 = 2621.7
        self.CO = 16.34
        self.NOx = 6.6
        self.SO2 = 20

        ## Battery
        self.SOC_min = 0.05
        self.SOC_max = 1
        self.SOC_initial = 1
        self.self_discharge_rate = 0     # Hourly self-discharge rate
        self.L_B = 7.5  # Life time (year)
        self.RT_B = ceil(self.n / self.L_B) - 1  # Replacement time

        # Lead Acid Battery
        self.Cnom_Leadacid = 83.4  # Li-ion nominal capacity [Ah]
        self.alfa_battery_leadacid = 1       # is the storage's maximum charge rate [A/Ah]
        self.c = 0.403              # the storage capacity ratio [unitless]
        self.k = 0.827              # the storage rate constant [1/h]
        self.Ich_max_leadacid = 16.7            # the storage's maximum charge current [A]
        self.Vnom_leadacid = 12              # the storage's nominal voltage [V]
        self.ef_bat_leadacid = 0.8           # Round trip efficiency
        self.Q_lifetime_leadacid = 8000  # Throughout in kWh

        # Li-ion Battery
        self.Ich_max_Li_ion = 167 # the storage's maximum charge current [A]
        self.Idch_max_Li_ion = 500  # the storage's maximum discharge current [A]
        self.alfa_battery_Li_ion = 1  # is the storage's maximum charge rate [A/Ah]
        self.Vnom_Li_ion = 6  # the storage's nominal voltage [V]
        self.Cnom_Li = 167 # Li-ion nominal capacity [Ah]
        self.ef_bat_Li = 0.90  # Round trip efficiency
        self.Q_lifetime_Li = 3000  # Throughout in kWh

        # Charger
        self.L_CH = 25      # Life time (year)
        self.RT_CH = ceil(self.n/self.L_CH) - 1    # Replacement time

        # Rated capacity
        self.Ppv_r = 0.5  # PV module rated power (kW)
        self.Pwt_r = 1  # WT rated power (kW)
        if self.Lead_acid == 1:
            self.Cbt_r = (self.Vnom_leadacid * self.Cnom_Leadacid) / 1000  # Battery rated Capacity (kWh)
        if self.Li_ion == 1:
            self.Cbt_r = (self.Vnom_Li_ion * self.Cnom_Li) / 1000  # Battery rated Capacity (kWh)
        self.Cdg_r = 0.5  # DG rated Capacity (kW)


        # Economic Parameters
        self.n_ir_rate = 5.5            # Nominal discount rate
        self.n_ir = self.n_ir_rate / 100
        self.e_ir_rate = 2              # Expected inflation rate
        self.e_ir = self.e_ir_rate / 100

        self.ir = (self.n_ir - self.e_ir) / (1 + self.e_ir)  # real discount rate

        self.Budget = 200e3   # Limit On Total Capital Cost

        self.Tax_rate = 0                 # Equipment sale tax Percentage
        self.System_Tax = self.Tax_rate / 100

        self.RE_incentives_rate = 0  # Federal tax credit percentage
        self.RE_incentives = self.RE_incentives_rate / 100

        # Pricing method
        self.Pricing_method = 2  # 1=Top down 2=bottom up

        # Top-down price definition
        if self.Pricing_method == 1:
            Total_PV_price = 2950

            from top_down import top_down_pricing
            self.Engineering_Costs, self.C_PV, self.R_PV, self.C_I, self.R_I, self.r_Sales_tax = top_down_pricing(Total_PV_price)

            # PV
            # self.R_PV_adj_rate = 0                            # PV module replacement cost yearly esclation rate (if positive) and reduction rate (if negative)
            # self.R_PV_adj = self.R_PV_adj_rate / 100
            self.MO_PV = 28.12 * (1 + self.r_Sales_tax)      # PV O&M cost ($/year/kw)

            # Inverter
            self.MO_I = 3 * (1 + self.r_Sales_tax)         # Inverter O&M cost ($/kW.year)

            # WT
            self.C_WT = 1200 * (1 + self.r_Sales_tax)      # Capital cost ($) per KW
            self.R_WT = 1200 * (1 + self.r_Sales_tax)      # Replacement Cost of WT Per KW
            self.MO_WT = 40 * (1 + self.r_Sales_tax)      # O&M cost ($/year/kw)

            # Diesel generator
            self.C_DG = 240.45 * (1 + self.r_Sales_tax)       # Capital cost ($/kW)
            self.R_DG = 240.45 * (1 + self.r_Sales_tax)       # Replacement Cost ($/kW)
            self.MO_DG = 0.064 * (1 + self.r_Sales_tax)     # O&M+ running cost ($/op.h)
            self.C_fuel = 1.39 * (1 + self.r_Sales_tax)             # Fuel Cost ($/L)
            self.C_fuel_adj_rate = 0                                # DG fuel cost yearly esclation rate (if positive) and reduction rate (if negative)
            self.C_fuel_adj = self.C_fuel_adj_rate / 100

            # Battery
            self.C_B = 458.06 * (1 + self.r_Sales_tax)              # Capital cost ($/kWh)
            self.R_B = 458.06 * (1 + self.r_Sales_tax)              # Replacement Cost ($/kWh)
            self.MO_B = 10 * (1 + self.r_Sales_tax)                   # Maintenance cost ($/kWh.year)

            # Charger
            self.C_CH = 149.99 * (1 + self.r_Sales_tax)  # Capital Cost ($)
            self.R_CH = 149.99 * (1 + self.r_Sales_tax)  # Replacement Cost ($)
            self.MO_CH = 0 * (1 + self.r_Sales_tax)   # O&M cost ($/year)

        else:
        ####### Pricing method 2=bottom up
        # Engineering Costs (Per/kW)
            self.Installation_cost = 160
            self.Overhead = 260
            self.Sales_and_marketing = 400
            self.Permiting_and_Inspection = 210
            self.Electrical_BoS = 370
            self.Structrual_BoS = 160
            self.Supply_Chain_costs = 0
            self.Profit_costs = 340
            self.Sales_tax = 80
            #self.Engineering_Costs = (self.Sales_tax + self.Profit_costs + self.Installation_cost + self.Overhead + self.Sales_and_marketing + self.Permiting_and_Inspection + self.Electrical_BoS + self.Structrual_BoS + self.Supply_Chain_costs)

            self.Engineering_Costs=0

            # PV
            self.C_PV = 2510             # Capital cost ($) per KW
            self.R_PV = 2510             # Replacement Cost of PV modules Per KW
            self.MO_PV = 28.88      # O&M cost ($/year/kw)

            # Inverter
            self.C_I = 440                 # Capital cost ($/kW)
            self.R_I = 440                  # Replacement cost ($/kW)
            self.MO_I = 3.08                    # O&M cost ($/kw.year)

            # WT
            self.C_WT = 1200      # Capital cost ($) per KW
            self.R_WT = 1200      # Replacement Cost of WT Per KW
            self.MO_WT = 40      # O&M cost ($/year/kw)

            # Diesel generator
            self.C_DG = 240.45       # Capital cost ($/KW)
            self.R_DG = 240.45       # Replacement Cost ($/kW)
            self.MO_DG = 0.066    # O&M+ running cost ($/op.h)
            self.C_fuel = 1.428  # Fuel Cost ($/L)
            self.C_fuel_adj_rate = 0  # DG fuel cost yearly escalation  rate (if positive) and reduction rate (if negative)
            self.C_fuel_adj = self.C_fuel_adj_rate / 100

            # Battery
            self.C_B = 458.06              # Capital cost ($/KWh)
            self.R_B = 458.06              # Replacement Cost ($/kW)
            self.MO_B = 10.27                # Maintenance cost ($/kw.year)

            # Charger
            self.C_CH = 0 # Capital Cost ($)
            self.R_CH = 0  # Replacement Cost ($)
            self.MO_CH = 0  # O&M cost ($/year)

        # Prices for Utility
        # Definition for the Utility Structures
        # 1 = flat rate
        # 2 = seasonal rate
        # 3 = monthly rate
        # 4 = tiered rate
        # 5 = seasonal tiered rate
        # 6 = monthly tiered rate
        # 7 = time of use rate


        # Hourly Rate Structure
        self.rateStructure = 7

        # Fixed expenses
        self.Annual_expenses = 0 # Annual expenses in $ for grid if any

        self.Grid_sale_tax_rate = 0 # Sale tax percentage of grid electricity

        self.Grid_Tax = self.Grid_sale_tax_rate / 100

        self.Grid_Tax_amount = 0 # Grid tax in kWh if any

        self.Grid_escalation_rate = 0 # Yearly escalation rate in grid electricity prices

        self.Grid_escalation = self.Grid_escalation_rate / 100

        self.Grid_credit = 0 # Credits offered by grid to users in $

        self.NEM_fee = 0 # Net metering one time setup fee

        # Monthly fixed charge structure
        self.Monthly_fixed_charge_system = 1

        if self.Monthly_fixed_charge_system == 1:  # Flat
            self.SC_flat = 9.95
            self.Service_charge = np.ones(12) * self.SC_flat
        else:  # Tiered
            self.SC_1 = 2.30  # tier 1 service charge
            self.Limit_SC_1 = 350  # limit for tier 1
            self.SC_2 = 7.9  # tier 2 service charge
            self.Limit_SC_2 = 1050  # limit for tier 2
            self.SC_3 = 22.70  # tier 3 service charge
            self.Limit_SC_3 = 1501  # limit for tier 3
            self.SC_4 = 22.70  # tier 4 service charge

            from service_charge import service_charge
            self.Service_charge = service_charge(self.daysInMonth, self.Eload_Previous, self.Limit_SC_1, self.SC_1, self.Limit_SC_2, self.SC_2, self.Limit_SC_3, self.SC_3, self.SC_4)

        # Hourly charges
        if self.rateStructure == 1:  # Flat rate
            self.flatPrice = 0.191

            from calcFlatRate import calcFlatRate
            self.Cbuy = calcFlatRate(self.flatPrice)

        elif self.rateStructure == 2:  # Seasonal rate
            self.seasonalPrices = np.array([0.28, 0.13])  # [summer, winter]
            self.season = np.array([0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0])  # define summer season 1= Summer

            from calcSeasonalRate import calcSeasonalRate
            self.Cbuy = calcSeasonalRate(self.seasonalPrices, self.season, self.daysInMonth)

        elif self.rateStructure == 3:  # Monthly rate
            self.monthlyPrices = np.array([0.54207, 0.53713, 0.38689, 0.30496, 0.28689, 0.28168, 0.30205, 0.28956, 0.26501, 0.26492, 0.3108, 0.40715])

            from calcMonthlyRate import calcMonthlyRate
            self.Cbuy = calcMonthlyRate(self.monthlyPrices, self.daysInMonth)

        elif self.rateStructure == 4:  # Tiered rate
            self.tieredPrices = np.array([0.1018, 0.1175, 0.1175])  # prices for tiers
            self.tierMax = np.array([300, 999999, 999999])  # max kWh limits for tiers

            from calcTieredRate import calcTieredRate
            self.Cbuy = calcTieredRate(self.tieredPrices, self.tierMax, self.Eload, self.daysInMonth)

        elif self.rateStructure == 5:  # Seasonal tiered rate
            self.seasonalTieredPrices = np.array([[0.077, 0.094, 0.094], [0.077, 0.094, 0.094]])  # [summer, winter][tier 1, 2, 3]
            self.seasonalTierMax = np.array([[600, 999999, 999999], [1000, 999999, 999999]])  # [summer, winter][tier 1, 2, 3] max kWh limits
            self.season = np.array([0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0])  # define summer season 1= Summer

            from calcSeasonalTieredRate import calcSeasonalTieredRate
            self.Cbuy = calcSeasonalTieredRate(self.seasonalTieredPrices, self.seasonalTierMax, self.Eload, self.season)

        elif self.rateStructure == 6:  # Monthly tiered rate
            self.monthlyTieredPrices = np.array([
                [0.077, 0.094, 0.094],
                [0.077, 0.094, 0.094],
                [0.077, 0.094, 0.094],
                [0.077, 0.094, 0.094],
                [0.077, 0.094, 0.094],
                [0.077, 0.094, 0.094],
                [0.077, 0.094, 0.094],
                [0.077, 0.094, 0.094],
                [0.077, 0.094, 0.094],
                [0.077, 0.094, 0.094],
                [0.077, 0.094, 0.094],
                [0.077, 0.094, 0.094]
            ])
            self.monthlyTierLimits = np.array([
                [1000, 999999, 999999],
                [1000, 999999, 999999],
                [1000, 999999, 999999],
                [1000, 999999, 999999],
                [600, 999999, 999999],
                [600, 999999, 999999],
                [600, 999999, 999999],
                [600, 999999, 999999],
                [600, 999999, 999999],
                [600, 999999, 999999],
                [1000, 999999, 999999],
                [1000, 999999, 999999]
            ])

            from calcMonthlyTieredRate import calcMonthlyTieredRate
            self.Cbuy = calcMonthlyTieredRate(self.monthlyTieredPrices, self.monthlyTierLimits, self.Eload)

        elif self.rateStructure == 7:  # Time of use rate
            self.onPrice = np.array([0.14, 0.14])  # prices for on-peak hours [summer, winter]
            self.midPrice = np.array([0.091, 0.091])   # prices for mid-peak hours [summer, winter]
            self.offPrice = np.array([0.065, 0.065])  # prices for off-peak hours [summer, winter]
            self.onHours = np.array([[11, 12, 13, 14, 15, 16], [7, 8, 9, 10, 17, 18]], dtype=object)  # on-peak hours [summer, winter]
            self.midHours = np.array([[7, 8, 9, 10, 17, 18], [11, 12, 13, 14, 15, 16]], dtype=object)  # mid-peak hours [summer, winter]
            self.season = np.array([0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0])  # define summer season 1= Summer
            self.holidays = np.array([1, 51, 97, 100, 142, 182, 219, 248, 273, 282, 315, 359, 360])  # Holidays based on the day in 365 days format

            from calcTouRate import calcTouRate
            self.Cbuy = calcTouRate(self.year, self.onPrice, self.midPrice, self.offPrice, self.onHours, self.midHours, self.season, self.daysInMonth, self.holidays)


        # Sell to the Grid
        self.sellStructure = 3

        if self.sellStructure == 1:
            self.Csell = np.full(8760, 0.049)

        elif self.sellStructure == 2:
            self.monthlysellprices = np.array([0.07054, 0.08169, 0.08452, 0.08748, 0.08788, 0.08510, 0.08158, 0.07903, 0.07683, 0.07203, 0.05783, 0.05878])

            from calcMonthlyRate import calcMonthlyRate
            self.Csell = calcMonthlyRate(self.monthlysellprices, self.daysInMonth)

        elif self.sellStructure == 3:
            self.Csell = self.Cbuy

        # Grid emission information
        # Emissions produced by Grid generators (kg/kWh)
        self.E_CO2 = 1.43
        self.E_SO2 = 0.01
        self.E_NOx = 0.39

        # Constraints for buying/selling from/to grid
        self.Pbuy_max = 6 # ceil(1.2 * max(self.Eload))  # kWh
        self.Psell_max = 200 # self.Pbuy_max

InData = Input_Data()
