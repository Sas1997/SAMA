import numpy as np
from math import ceil
import pandas as pd
from math import ceil

# Optimization
# PSO Parameters
class Input_Data:
    def __init__(self):
        self.MaxIt = 99  # Maximum Number of Iterations
        self.nPop = 100  # Population Size (Swarm Size)
        self.w = 1  # Inertia Weight
        self.wdamp = 0.99  # Inertia Weight Damping Ratio
        self.c1 = 2 # Personal Learning Coefficient
        self.c2 = 2  # Global Learning Coefficient

        # Multi-run
        self.Run_Time = 1  # Total number of runs in each click

        # Calendar
        self.n = 25  # Lifetime of system in simulations (years)
        self.year = 2023  # Specify the desired year
        # Days in each month
        # Determine the number of days in February based on whether it's a leap year
        if self.year % 4 == 0:
            self.daysInMonth = [31, 29, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
        else:
            self.daysInMonth = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]

        # Reading global inputs
        self.path = 'Data.csv'
        self.Data = pd.read_csv(self.path, header=None).values

        # Electrical load definitions
        # 1=Hourly load based on the actual data
        # 2=Monthly average load
        # 3=Annual average load
        load_type = 1

        if load_type == 1:
            self.Eload = np.array(self.Data[:, 0])
        elif load_type == 2:
            self.Monthly_average_load = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]  # Define the monthly averages for load here
            self.hCount = 0
            self.Eload = []
            for m in range(12):
                for h in range(24 * self.daysInMonth[m]):
                    self.Eload.append(self.Monthly_average_load[m])
        else:
            self.Annual_average_load = 1  # Define the annual average for load here
            self.Eload = np.full(8760, self.Annual_average_load)

        self.G = np.array(self.Data[:, 1])
        self.T = np.array(self.Data[:, 2])
        self.Vw = np.array(self.Data[:, 3])
        self.Eload_Previous = np.array(self.Data[:, 0])

        # Other inputs
        # Technical data
        # Type of system (1: included, 0=not included)
        self.PV = 1
        self.WT = 0
        self.DG = 1
        self.Bat = 1
        self.Grid = 1

        # Constraints
        self.LPSP_max_rate = 0.0999999  # Maximum loss of power supply probability
        self.LPSP_max = self.LPSP_max_rate / 100

        self.RE_min_rate = 75  # Minimum Renewable Energy Capacity
        self.RE_min = self.RE_min_rate / 100

        self.EM = 0  # 0: LCOE, 1:LCOE+LEM

        # Rated capacity
        self.Ppv_r = 0.5  # PV module rated power (kW)
        self.Pwt_r = 1  # WT rated power (kW)
        self.Cbt_r = 1  # Battery rated Capacity (kWh)
        self.Cdg_r = 0.5  # DG rated Capacity (kW)

        # PV
        self.fpv = 0.9       # the PV derating factor [%]
        self.Tcof = 0        # temperature coefficient [%/C]
        self.Tref = 25       # temperature at standard test condition
        self.Tc_noct = 45    # Nominal operating cell temperature
        self.Ta_noct = 20
        self.G_noct = 800
        self.gama = 0.9
        self.n_PV = 0.2182   # Efficiency of PV module
        self.Gref = 1000     # 1000 W/m^2
        self.L_PV = 25       # Life time (year)
        self.RT_PV = ceil(self.n/self.L_PV) - 1   # Replecement time

        # Inverter
        self.n_I = 0.96      # Efficiency
        self.L_I = 25        # Life time (year)
        self.DC_AC_ratio = 1.99 # Maximum acceptable DC to AC ratio
        self.RT_I = ceil(self.n/self.L_I) - 1    # Replecement time

        # WT data
        self.h_hub = 17              # Hub height
        self.h0 = 43.6               # anemometer height
        self.nw = 1                  # Electrical Efficiency
        self.v_cut_out = 25          # cut out speed
        self.v_cut_in = 2.5          # cut in speed
        self.v_rated = 9.5           # rated speed(m/s)
        self.alfa_wind_turbine = 0.14    # coefficient of friction (0.11 for extreme wind conditions, and 0.20 for normal wind conditions)
        self.L_WT = 20           # Life time (year)
        self.RT_WT = ceil(self.n/self.L_WT) - 1    # Replecement time

        # Diesel generator
        self.LR_DG = 0.25        # Minimum Load Ratio (%)
        # Diesel Generator fuel curve
        self.a = 0.2730          # L/hr/kW output
        self.b = 0.0330          # L/hr/kW rated
        self.TL_DG = 24000       # Life time (h)
        # Emissions produced by Diesel generator for each fuel in littre [L] g/L
        self.CO2 = 2621.7
        self.CO = 16.34
        self.NOx = 6.6
        self.SO2 = 20

        # Battery
        self.SOC_min = 0.2
        self.SOC_max = 1
        self.SOC_initial = 0.5
        self.Q_lifetime = 8000       # Throughout in kWh
        self.self_discharge_rate = 0     # Hourly self-discharge rate
        self.alfa_battery = 1       # is the storage's maximum charge rate [A/Ah]
        self.c = 0.403              # the storage capacity ratio [unitless]
        self.k = 0.827              # the storage rate constant [h-1]
        self.Imax = 16.7            # the storage's maximum charge current [A]
        self.Vnom = 12              # the storage's nominal voltage [V]
        self.ef_bat = 0.8           # Round trip efficiency
        self.L_B = 7.5              # Life time (year)
        self.RT_B = ceil(self.n/self.L_B) - 1    # Replacement time

        # Charger
        self.L_CH = 25      # Life time (year)
        self.RT_CH = ceil(self.n/self.L_CH) - 1    # Replacement time

        # Economic Parameters
        self.n_ir_rate = 4.5             # Nominal discount rate
        self.n_ir = self.n_ir_rate / 100
        self.e_ir_rate = 2               # Expected inflation rate
        self.e_ir = self.e_ir_rate / 100

        self.ir = (self.n_ir - self.e_ir) / (1 + self.e_ir)  # real discount rate

        self.Budget = 200e3   # Limit On Total Capital Cost

        self.Tax_rate = 0                 # Equipment sale tax Percentage
        self.System_Tax = self.Tax_rate / 100

        self.RE_incentives_rate = 30  # Federal tax credit percentage
        self.RE_incentives = self.RE_incentives_rate / 100

        # Pricing method
        self.Pricing_method = 1  # 1=Top down 2=bottom up

        # Top-down price definition
        if self.Pricing_method == 1:
            # Pricing method 1/top down
            self.Total_PV_price = 2950
            # NREL percentages
            self.r_PV = 0.1812
            self.r_inverter = 0.1492
            self.r_Installation_cost = 0.0542
            self.r_Overhead = 0.0881
            self.r_Sales_and_marketing = 0.1356
            self.r_Permiting_and_Inspection = 0.0712
            self.r_Electrical_BoS = 0.1254
            self.r_Structrual_BoS = 0.0542
            self.r_Profit_costs = 0.1152
            self.r_Sales_tax = 0.0271
            self.r_Supply_Chain_costs = 0

            # Engineering Costs (Per/kW)
            self.Installation_cost = self.Total_PV_price * self.r_Installation_cost
            self.Overhead = self.Total_PV_price * self.r_Overhead
            self.Sales_and_marketing = self.Total_PV_price * self.r_Sales_and_marketing
            self.Permiting_and_Inspection = self.Total_PV_price * self.r_Permiting_and_Inspection
            self.Electrical_BoS = self.Total_PV_price * self.r_Electrical_BoS
            self.Structrual_BoS = self.Total_PV_price * self.r_Structrual_BoS
            self.Profit_costs = self.Total_PV_price * self.r_Profit_costs
            self.Sales_tax = self.Total_PV_price * self.r_Sales_tax
            self.Supply_Chain_costs = self.Total_PV_price * self.r_Supply_Chain_costs
            self.Engineering_Costs = (self.Sales_tax + self.Profit_costs + self.Installation_cost + self.Overhead + self.Sales_and_marketing +
                                 self.Permiting_and_Inspection + self.Electrical_BoS + self.Structrual_BoS + self.Supply_Chain_costs)

            # PV
            self.C_PV = self.Total_PV_price * self.r_PV         # Capital cost ($) per KW
            self.R_PV = self.Total_PV_price * self.r_PV        # Replacement Cost of PV modules Per KW
            self.MO_PV = 28.12 * (1 + self.r_Sales_tax)      # PV O&M cost ($/year/kw)

            # Inverter
            self.C_I = self.Total_PV_price * self.r_inverter         # Capital cost ($/kW)
            self.R_I = self.Total_PV_price * self.r_inverter         # Replacement cost ($/kW)
            self.MO_I = 3 * (1 + self.r_Sales_tax)         # Inverter O&M cost ($/kW.year)

            # WT
            self.C_WT = 1200 * (1 + self.r_Sales_tax)      # Capital cost ($) per KW
            self.R_WT = 1200 * (1 + self.r_Sales_tax)      # Replacement Cost of WT Per KW
            self.MO_WT = 40 * (1 + self.r_Sales_tax)      # O&M cost ($/year/kw)

            # Diesel generator
            self.C_DG = 240.45 * (1 + self.r_Sales_tax)       # Capital cost ($/KW)
            self.R_DG = 240.45 * (1 + self.r_Sales_tax)       # Replacement Cost ($/kW)
            self.MO_DG = 0.064 * (1 + self.r_Sales_tax)     # O&M+ running cost ($/op.h)
            self.C_fuel = 1.39 * (1 + self.r_Sales_tax)             # Fuel Cost ($/L)

            # Battery
            self.C_B = 458.06 * (1 + self.r_Sales_tax)              # Capital cost ($/kWh)
            self.R_B = 458.06 * (1 + self.r_Sales_tax)              # Replacement Cost ($/kW)
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
            self.Engineering_Costs = (self.Sales_tax + self.Profit_costs + self.Installation_cost + self.Overhead + self.Sales_and_marketing +
                                  self.Permiting_and_Inspection + self.Electrical_BoS + self.Structrual_BoS + self.Supply_Chain_costs)

            #self.Engineering_Costs=0

            # PV
            self.C_PV = 2510             # Capital cost ($) per KW
            self.R_PV = 2510             # Replacement Cost of PV modules Per KW
            self.MO_PV = 28.12      # O&M cost ($/year/kw)

            # Inverter
            self.C_I = 440                 # Capital cost ($/kW)
            self.R_I = 440                  # Replacement cost ($/kW)
            self.MO_I = 3                    # O&M cost ($/kw.year)

            # WT
            self.C_WT = 1200      # Capital cost ($) per KW
            self.R_WT = 1200      # Replacement Cost of WT Per KW
            self.MO_WT = 40      # O&M cost ($/year/kw)

            # Diesel generator
            self.C_DG = 240.45       # Capital cost ($/KW)
            self.R_DG = 240.45       # Replacement Cost ($/kW)
            self.MO_DG = 0.064    # O&M+ running cost ($/op.h)
            self.C_fuel = 1.39  # Fuel Cost ($/L)

            # Battery
            self.C_B = 458.06              # Capital cost ($/KWh)
            self.R_B = 458.06              # Replacement Cost ($/kW)
            self.MO_B = 10                # Maintenance cost ($/kw.year)

            # Charger
            self.C_CH = 149.99 * (1 + self.r_Sales_tax)  # Capital Cost ($)
            self.R_CH = 149.99 * (1 + self.r_Sales_tax)  # Replacement Cost ($)
            self.MO_CH = 0 * (1 + self.r_Sales_tax)  # O&M cost ($/year)

        # Prices for Utility
        # Definition for the Utility Structures
        # 1 = flat rate
        # 2 = seasonal rate
        # 3 = monthly rate
        # 4 = tiered rate
        # 5 = seasonal tiered rate
        # 6 = monthly tiered rate
        # 7 = time of use rate

        # Grid emission information
        # Emissions produced by Grid generators (g/kW)
        self.E_CO2 = 1.43
        self.E_SO2 = 0.01
        self.E_NOx = 0.39


        # Rate Structure
        self.rateStructure = 4

        # Fixed expenses
        self.Annual_expenses = 0
        self.Grid_sale_tax_rate = 7
        self.Grid_Tax = self.Grid_sale_tax_rate / 100

        # Monthly fixed charge
        self.Monthly_fixed_charge_system = 1

        if self.Monthly_fixed_charge_system == 1:  # Flat
            self.SC_flat = 9.95
            self.Service_charge = np.ones(12) * self.SC_flat
        else:  # Tiered
            self.SC_1 = 2.30  # tier 1 service charge
            self.Limit_SC_1 = 500  # limit for tier 1
            self.SC_2 = 7.9  # tier 2 service charge
            self.Limit_SC_2 = 1500  # limit for tier 2
            self.SC_3 = 22.7  # tier 3 service charge
            self.Limit_SC_3 = 1500  # limit for tier 3
            self.SC_4 = 22.7  # tier 4 service charge
            self.totalmonthlyload = np.zeros((12, 1))
            self.hourCount = 0
            for m in range(12):
                self.monthlyLoad = 0
                for h in range(24 * self.daysInMonth[m]):
                    self.monthlyLoad += self.Eload_Previous[self.hourCount]
                    self.hourCount += 1
                self.totalmonthlyload[m, 0] = self.monthlyLoad

            self.max_monthly_load = max(self.totalmonthlyload)
            if self.max_monthly_load < self.Limit_SC_1:
                self.Service_charge = np.ones(12) * self.SC_1
            elif self.max_monthly_load < self.Limit_SC_2:
                self.Service_charge = np.ones(12) * self.SC_2
            elif self.max_monthly_load < self.Limit_SC_3:
                self.Service_charge = np.ones(12) * self.SC_3
            else:
                self.Service_charge = np.ones(12) * self.SC_4

        # Hourly charges
        if self.rateStructure == 1:  # Flat rate
            self.flatPrice = 0.18035
            from calcFlatRate import calcFlatRate
            self.Cbuy = calcFlatRate(self.flatPrice)

        elif self.rateStructure == 2:  # Seasonal rate
            self.seasonalPrices = np.array([0.17, 0.13])  # [summer, winter]
            self.season = np.array([0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0])  # define summer season 1= Summer
            from calcSeasonalRate import calcSeasonalRate
            self.Cbuy = calcSeasonalRate(self.seasonalPrices, self.season, self.daysInMonth)

        elif self.rateStructure == 3:  # Monthly rate
            self.monthlyPrices = np.array([0.15, 0.14, 0.13, 0.16, 0.11, 0.10, 0.12, 0.13, 0.14, 0.10, 0.15, 0.16])
            from calcMonthlyRate import calcMonthlyRate
            self.Cbuy = calcMonthlyRate(self.monthlyPrices, self.daysInMonth)

        elif self.rateStructure == 4:  # Tiered rate
            self.tieredPrices = np.array([0.1018, 0.1175, 0.1175])  # prices for tiers
            self.tierMax = np.array([300, 999999, 999999])  # max kWh limits for tiers
            from calcTieredRate import calcTieredRate
            self.Cbuy = calcTieredRate(self.tieredPrices, self.tierMax, self.Eload, self.daysInMonth)

        elif self.rateStructure == 5:  # Seasonal tiered rate
            self.seasonalTieredPrices = np.array([[0.05, 0.08, 0.14], [0.09, 0.13, 0.2]])  # [summer, winter][tier 1, 2, 3]
            self.seasonalTierMax = np.array([[400, 800, 4000], [1000, 1500, 4000]])  # [summer, winter][tier 1, 2, 3] max kWh limits
            self.season = np.array([0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0])  # define summer season 1= Summer
            from calcSeasonalTieredRate import calcSeasonalTieredRate
            self.Cbuy = calcSeasonalTieredRate(self.seasonalTieredPrices, self.seasonalTierMax, self.Eload, self.season)

        elif self.rateStructure == 6:  # Monthly tiered rate
            self.monthlyTieredPrices = np.array([
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
                [0.19192, 0.25051, 0.25051]
            ])
            self.monthlyTierLimits = np.array([
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
                [500, 1500, 1501]
            ])
            from calcMonthlyTieredRate import calcMonthlyTieredRate
            self.Cbuy = calcMonthlyTieredRate(self.monthlyTieredPrices, self.monthlyTierLimits, self.Eload)

        elif self.rateStructure == 7:  # Time of use rate
            self.onPrice = np.array([0.3279, 0.1547])  # prices for on-peak hours [summer, winter]
            self.midPrice = np.array([0.1864, 0.1864])   # prices for mid-peak hours [summer, winter]
            self.offPrice = np.array([0.1350, 0.1120])  # prices for off-peak hours [summer, winter]
            self.onHours = np.array([[17, 18, 19], [17, 18, 19]], dtype=object)  # on-peak hours [summer, winter]
            self.midHours = np.array([[13, 14, 15, 16, 20, 21, 22, 23, 24], []], dtype=object)  # mid-peak hours [summer, winter]
            self.season = np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0])  # define summer season 1= Summer
            self.holidays = np.array([1, 16, 51, 149, 170, 185, 247, 282, 315, 327, 359])  # Holidays based on the day in 365 days format
            from calcTouRate import calcTouRate
            self.Cbuy = calcTouRate(self.year, self.onPrice, self.midPrice, self.offPrice, self.onHours, self.midHours, self.season, self.daysInMonth, self.holidays)

        #Saving the Cbuy values
        Cbuy_df = pd.DataFrame(self.Cbuy, columns=['Column_Name'])
        Cbuy_df.index = Cbuy_df.index + 1
        Cbuy_df = Cbuy_df.reset_index(drop=True)
        Cbuy_df.to_excel('Cbuy.xlsx', header=False, index=False)

        self.Csell = 0.0487

        ## Emissions produced by Grid generators (g/kW)
        self.E_CO2 = 1.43
        self.E_SO2 = 0.01
        self.E_NOx = 0.39

        # Constraints for buying/selling from/to grid
        self.Pbuy_max = 6 # ceil(1.2 * max(self.Eload))  # kWh
        self.Psell_max = 200 # self.Pbuy_max

InData = Input_Data()
