import numpy as np
import pandas as pd
from math import ceil
from daysInMonth import daysInMonth

# Optimization
# PSO Parameters
class Input_Data:
    def __init__(self):
        self.Cash_Flow_adv = 0
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
        self.holidays = np.array([1, 51, 97, 100, 142, 182, 219, 248, 273, 282, 315, 359, 360])  # Holidays based on the day in 365 days format
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
            path_Eload = 'content/Eload.csv'
            EloadData = pd.read_csv(path_Eload, header=None).values
            self.Eload_eh = np.array(EloadData[:, 0])

        elif load_type == 2:
            Monthly_haverage_load = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12])  # Define the monthly hourly averages for load here

            from dataextender import dataextender
            self.Eload_eh = dataextender(self.daysInMonth, Monthly_haverage_load)

        elif load_type == 3:
            Monthly_daverage_load = np.array([10, 20, 31, 14, 15, 16, 17, 18, 19, 10, 11, 12])  # Define the monthly daily averages for load here

            Monthly_haverage_load = Monthly_daverage_load / 24
            from dataextender import dataextender
            self.Eload_eh = dataextender(self.daysInMonth, Monthly_haverage_load)

        elif load_type == 4:
            Monthly_total_load = np.array([321, 223, 343, 423, 544, 623, 237, 843, 239, 140, 121, 312])  # Define the monthly total load here

            Monthly_haverage_load = Monthly_total_load / (self.daysInMonth * 24)
            from dataextender import dataextender
            self.Eload_eh = dataextender(self.daysInMonth, Monthly_haverage_load)

        elif load_type == 5: # Based on the generic load
            peak_month = 'July'
            user_defined_load = np.array([300, 350, 320, 320, 320, 320, 320, 320, 320, 320, 320, 320]) # Define the monthly total load here

            from generic_load import generic_load
            self.Eload_eh = generic_load(load_type, 1, peak_month, self.daysInMonth, user_defined_load)

        elif load_type == 6:
            Annual_haverage_load = 1  # Define the annual hourly average for load here

            self.Eload_eh = np.full(8760, Annual_haverage_load)

        elif load_type == 7:
            Annual_daverage_load = 10  # Define the annual hourly average for load here

            Annual_haverage_load = Annual_daverage_load / 24
            self.Eload_eh = np.full(8760, Annual_haverage_load)

        elif load_type == 8: #Annual total load
            Annual_total_load = 9500  # Define the annual hourly average for load here
            peak_month = 'July'

            from generic_load import generic_load
            self.Eload_eh = generic_load(load_type, 1, peak_month, self.daysInMonth, Annual_total_load)

        elif load_type == 9:
            peak_month = 'July'

            from generic_load import generic_load
            self.Eload_eh = generic_load(load_type, 1, peak_month, self.daysInMonth, 1)

        elif load_type == 10:
            self.path_Eload_daily = 'content/Eload_daily.csv'
            self.EloadData_daily = pd.read_csv(self.path_Eload_daily, header=None).values
            self.user_defined_load = np.array(self.EloadData_daily[:, 0])
            peak_month = 'July'

            from generic_load import generic_load
            self.Eload_eh = generic_load(load_type, 1, peak_month, self.daysInMonth, self.user_defined_load)


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

            Eload_eh_Previous = self.Eload_eh

        elif load_previous_year_type == 2:

            path_Eload_Previous = 'content/Eload_previousyear.csv'
            Eload_PreviousData = pd.read_csv(path_Eload_Previous, header=None).values
            Eload_eh_Previous = np.array(Eload_PreviousData[:, 0])

        elif load_previous_year_type == 3:
            Monthly_haverage_load_previous = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12])  # Define the monthly hourly averages for load here

            from dataextender import dataextender
            Eload_eh_Previous = dataextender(self.daysInMonth, Monthly_haverage_load_previous)

        elif load_previous_year_type == 4:
            Monthly_daverage_load_previous = np.array([10, 20, 31, 14, 15, 16, 17, 18, 19, 10, 11, 12])  # Define the monthly hourly averages for load here
            Monthly_haverage_load_previous = Monthly_daverage_load_previous / 24

            from dataextender import dataextender
            Eload_eh_Previous = dataextender(self.daysInMonth, Monthly_haverage_load_previous)

        elif load_previous_year_type == 5:
            Monthly_total_load_previous = np.array([321, 223, 343, 423, 544, 623, 237, 843, 239, 140, 121, 312])  # Define the monthly total load here

            Monthly_haverage_load_previous = Monthly_total_load_previous / (self.daysInMonth * 24)
            from dataextender import dataextender
            Eload_eh_Previous = dataextender(self.daysInMonth, Monthly_haverage_load_previous)

        elif load_previous_year_type == 6:
            peak_month = 'July'
            user_defined_load = np.array([300, 350, 320, 320, 320, 320, 320, 320, 320, 320, 320, 320])  # Define the monthly total load here

            from generic_load import generic_load
            Eload_eh_Previous = generic_load(1, load_previous_year_type, peak_month, self.daysInMonth, user_defined_load)

        elif load_previous_year_type == 7:
            Annual_haverage_load_previous = 1  # Define the annual hourly average for load here

            Eload_eh_Previous = np.full(8760, Annual_haverage_load_previous)

        elif load_previous_year_type == 8:
            Annual_daverage_load_previous = 10  # Define the annual hourly average for load here

            Annual_haverage_load_previous = Annual_daverage_load_previous / 24
            Eload_eh_Previous = np.full(8760, Annual_haverage_load_previous)

        elif load_previous_year_type == 9: # Annual total load for previous year
            Annual_total_load = 9500  # Define the annual hourly average for load here
            peak_month = 'July'

            from generic_load import generic_load
            Eload_eh_Previous = generic_load(1, load_previous_year_type, peak_month, self.daysInMonth, Annual_total_load)

        elif load_previous_year_type == 10:
            peak_month = 'July'

            from generic_load import generic_load
            Eload_eh_Previous = generic_load(1, load_previous_year_type, peak_month, self.daysInMonth, 1)

        elif load_previous_year_type == 11:
            self.path_Eload_daily = 'content/Eload_daily.csv'
            self.EloadData_daily = pd.read_csv(self.path_Eload_daily, header=None).values
            self.user_defined_load = np.array(self.EloadData_daily[:, 0])
            peak_month = 'July'

            from generic_load import generic_load
            Eload_eh_Previous = generic_load(1, load_previous_year_type, peak_month, self.daysInMonth, self.user_defined_load)

        # Thermal load
        Tload_type = 1  # Determine the way you want to input the electrical load by choosing one of the numbers above

        if Tload_type == 1:
            data_2 = pd.read_excel('C:/Users/alisa/Desktop/FAST/SAMA V1.04-HP added/content/Heat Pump/house_load.xlsx')
            self.Hload = data_2.iloc[:, 1].to_numpy()  # hourly heating load [kW]
            self.Cload = data_2.iloc[:, 2].to_numpy()  # hourly cooling load [kW]



        # Irradiance definitions
        # 1=Hourly irradiance based on POA calculator
        # 2=Hourly POA irradiance based on the user CSV file
        weather_url = 'content/METEO.csv'
        data = pd.read_csv(weather_url, header=2)
        columns = data.columns

        if 'Pressure' in columns:
            self.P = data['Pressure']
        else:
            print("No exact match for 'Pressure' found.")


        azimuth = 180
        tilt = 33  # Tilt angle of PV modules
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
            self.Vw = dataextender(self.daysInMonth, self.Monthly_average_windspeed)

        else: # Annual average Wind speed

            self.Annual_average_windspeed = 10
            self.Vw = np.full(8760, self.Annual_average_windspeed)


        # Other inputs
        # Technical data
        # Type of system (1: included, 0=not included)
        self.PV = 1
        self.WT = 0
        self.DG = 1
        self.Bat = 1
        self.Lead_acid = 0
        self.Li_ion = 1
        self.Grid = 1
        # Net metering scheme
        # If compensation is towrds credits in Net metering, not monetary comenstation, by choosing the below option (putting NEM equals to 1), the yearly credits will be reconciled after 12 months
        self.NEM = 1

        # if there this a capacity or size limit such as NEM cap, choose 1 and put the value of cap in kW for self.cap_size, if not but the system must be sized to the customer’s recent annual load choose 2, if you want to size the system according to the rooftop size, choose 3, and if there is not a limit choose 4
        self.cap_option = 3

        self.cap_size = 0
        self.generation_cap = 150  # Generation capacity based on the consumption in percentage
        self.available_roof_surface = 80  # available roof space for solar in m2
        self.PVPanel_surface_per_rated_capacity = 5  # PV Roof Space Required in m2 per rated capacity of PV
        # self.PV_surface_ratio = self.available_roof_surface / self.PVPanel_surface_per_rated_capacity

        if self.Grid == 0:
            self.NEM = 0

        # Select the heatpump with 1
        self.HP = 0

        # Select the EV with 1
        self.EV = 0

        if self.HP == 1:
            from BB_HP import Heat_Pump_Model
            self.Eload_hp, _, _, _, _, _, _ = Heat_Pump_Model(self.T, self.P/10, self.Hload, self.Cload)
            self.Eload = self.Eload_eh + self.Eload_hp.to_numpy()
            self.Eload_Previous = Eload_eh_Previous + self.Eload_hp.to_numpy()
        else:
            self.Eload = self.Eload_eh
            self.Eload_Previous = Eload_eh_Previous
            self.Eload_hp = 0

        self.NG_Grid = 0


        data = {'Eload': self.Eload, 'Eload_eh': self.Eload_eh, 'Eload_hp': self.Eload_hp, 'G': self.G, 'T': self.T, 'Vw': self.Vw}
        df = pd.DataFrame(data)
        df.to_csv('output/data/Inputs.csv', index=False)

        # Constraints
        self.LPSP_max_rate = 0.000999  # Maximum loss of power supply probability percentage

        self.LPSP_max = self.LPSP_max_rate / 100

        self.RE_min_rate = 50  # Minimum Renewable Energy Capacity percentage

        self.RE_min = self.RE_min_rate / 100

        self.EM = 0  # 0: NPC, 1:NPC+LEM

        # PV
        self.fpv = 0.9       # the PV derating factor [%]
        self.Tcof = -0.3       # temperature coefficient [%/C]
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
        self.n_I = 0.96     # Efficiency
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
        self.a = 0.4388          # L/hr/kW output
        self.b = 0.1097         # L/hr/kW rated
        self.TL_DG = 24000       # Life time (h)

        # Emissions produced by Diesel generator for each fuel in litre [L] kg/L
        self.CO2 = 2.29
        self.CO = 0
        self.NOx = 0
        self.SO2 = 0

        ## Battery
        self.SOC_min = 0.1
        self.SOC_max = 1
        self.SOC_initial = 0.5
        self.self_discharge_rate = 0     # Hourly self-discharge rate
        self.L_B = 10  # Life time (year)
        self.RT_B = ceil(self.n / self.L_B) - 1  # Replacement time

        # Lead Acid Battery
        self.Cnom_Leadacid = 83.4  # Lead Acid nominal capacity [Ah]
        self.alfa_battery_leadacid = 1       # is the storage's maximum charge rate [A/Ah]
        self.c = 0.403              # the storage capacity ratio [unitless]
        self.k_lead_acid = 0.827              # the storage rate constant [1/h]
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

        # Heat Pump

        self.L_HP = 5       # Life time (year)
        self.RT_HP = ceil(self.n / self.L_HP) - 1  # Replacement time

        # Rated capacity
        self.Ppv_r = 1  # PV module rated power (kW)
        self.Pwt_r = 1  # WT rated power (kW)
        if self.Lead_acid == 1:
            self.Cbt_r = (self.Vnom_leadacid * self.Cnom_Leadacid) / 1000  # Battery rated Capacity (kWh)
        if self.Li_ion == 1:
            self.Cbt_r = (self.Vnom_Li_ion * self.Cnom_Li) / 1000  # Battery rated Capacity (kWh)
        self.Cdg_r = 5.5  # DG rated Capacity (kW)
        self.Php_r = 1000 # Rated size of HP (BTU/hr)

        # EV Data
        self.Tin = 17  # Arrival time to home
        self.Tout = 8  # Departure time from home

        self.C_ev = 82  # EV Battery Capacity (kWh)
        self.SOCe_min = 0.03  # Minimum SOC
        self.SOCe_max = 0.97  # Maximum SOC
        self.C_ev_usable = self.C_ev * (self.SOCe_max - self.SOCe_min)
        self.SOCe_initial = 0.85
        self.Pev_max = 11  # EV Maximum charge and discharge rate
        self.Range_EV = 468  # in km EV travel range with one full charge
        self.Daily_trip = 68 # in kW, average amount daily you travel
        self.SOC_dep = 0.85  # SOC at departure time
        self.SOC_arr = self.SOC_dep - ((self.Daily_trip * self.C_ev_usable) / (self.Range_EV * self.C_ev))  # SOC at arrival time
        self.n_e = 0.9  # EV battery charge efficiency
        self.self_discharge_rate_ev = 0
        self.L_EV_dis = 400000 # in km lifetime of EV battery in terms of total distance driven
        self.degradation_percent = 0.019 # Degradation percentage per step km
        self.step_km = 1000
        from Ev_Battery_Throughput import calculate_ev_battery_throughput
        _, self.Q_lifetime_ev = calculate_ev_battery_throughput(self.C_ev_usable, self.degradation_percent, self.L_EV_dis, self.Range_EV, self.step_km)  #Lifetime Throughput
        #print(self.Q_lifetime_ev)
        self.L_EV = 25 #np.floor(self.L_EV_dis/(self.Daily_trip * np.sum(self.daysInMonth)))
        self.RT_EV = ceil(self.n / self.L_EV) - 1  # Replacement time

        self.treat_special_days_as_home = False
        from EV_Presence import determine_EV_presence
        self.EV_p = determine_EV_presence(self.year, self.Tout, self.Tin, self.holidays, self.treat_special_days_as_home)

        # # EV = 1 indicates the presence of EV at home
        # self.EV_p = np.ones(365 * 24, dtype=int)
        #
        # for d in range(1, 366):  # Loop over days (1 to 366)
        #     if (d - 1) % 7 > 1:  # Weekdays except weekends (Saturday and Sunday)
        #         tt = np.arange(24 * (d - 1), 24 * d)  # Hours of the day
        #         self.EV_p[tt[self.Tout:self.Tin]] = 0  # EV is not at home during these hours


        # Economic Parameters
        self.n_ir_rate = 4.5            # Nominal discount rate
        self.n_ir = self.n_ir_rate / 100
        self.e_ir_rate = 2              # Expected inflation rate
        self.e_ir = self.e_ir_rate / 100

        self.ir = (self.n_ir - self.e_ir) / (1 + self.e_ir)  # real discount rate

        self.Budget = 200e4   # Limit On Total Capital Cost

        self.Tax_rate = 0                 # Equipment sale tax Percentage
        self.System_Tax = self.Tax_rate / 100

        self.RE_incentives_rate = 30  # Federal tax credit percentage
        self.RE_incentives = self.RE_incentives_rate / 100

        # Pricing method
        self.Pricing_method = 2  # 1=Top down 2=bottom up

        # Top-down price definition
        if self.Pricing_method == 1:
            Total_PV_price = 2111

            from top_down import top_down_pricing
            self.Engineering_Costs, self.C_PV, self.R_PV, self.C_I, self.R_I = top_down_pricing(Total_PV_price)

            # PV
            # self.R_PV_adj_rate = 0                            # PV module replacement cost yearly esclation rate (if positive) and reduction rate (if negative)
            # self.R_PV_adj = self.R_PV_adj_rate / 100
            self.MO_PV = 30.36      # PV O&M cost ($/year/kw)

            # Inverter
            self.MO_I = 0        # Inverter O&M cost ($/kW.year)

            # WT
            self.C_WT = 1200      # Capital cost ($) per KW
            self.R_WT = 1200      # Replacement Cost of WT Per KW
            self.MO_WT = 40      # O&M cost ($/year/kw)

            # Diesel generator
            self.C_DG = 818      # Capital cost ($/kW)
            self.R_DG = 818       # Replacement Cost ($/kW)
            self.MO_DG = 0.016      # O&M+ running cost ($/op.h)
            self.C_fuel = 1.281            # Fuel Cost ($/L)
            self.C_fuel_adj_rate = 2                                # DG fuel cost yearly esclation rate (if positive) and reduction rate (if negative)
            self.C_fuel_adj = self.C_fuel_adj_rate / 100

            # Battery
            self.C_B = 1450              # Capital cost ($/kWh)
            self.R_B = 1450             # Replacement Cost ($/kWh)
            self.MO_B = 10                 # Maintenance cost ($/kWh.year)

            # Charger
            self.C_CH = 149.99   # Capital Cost ($)
            self.R_CH = 149.99   # Replacement Cost ($)
            self.MO_CH = 0   # O&M cost ($/year)

            # Heat Pump
            self.C_HP = 109.5 # Capital Cost ($)
            self.R_HP = 109.5   # Replacement Cost ($)
            self.MO_HP = 20  # O&M cost ($/year)

            # EV
            self.Cost_EV = 0
            self.R_EVB = 9840  # Replacement cost for EV battery full pack($)
            self.MO_EV = 0  # O&M cost ($/year)

        else:
        ####### Pricing method 2=bottom up
        # Engineering Costs (Per/kW)
        #     self.Fieldwork = 178
        #     self.Officework = 696
        #     self.Other = 586
        #     self.Permiting_and_Inspection = 0
        #     self.Electrical_BoS = 333
        #     self.Structrual_BoS = 237
        #     self.Supply_Chain_costs = 0
        #     self.Profit_costs = 0
        #     self.Sales_tax = 0
            self.Engineering_Costs = 2030 #(self.Sales_tax + self.Profit_costs + self.Fieldwork + self.Officework + self.Other + self.Permiting_and_Inspection + self.Electrical_BoS + self.Structrual_BoS + self.Supply_Chain_costs)

            #self.Engineering_Costs = 0

            # PV
            self.C_PV = 338            # Capital cost ($) per KW
            self.R_PV = 338           # Replacement Cost of PV modules Per KW
            self.MO_PV = 30.36      # O&M cost ($/year/kw)

            # Inverter
            self.C_I = 314                 # Capital cost ($/kW)
            self.R_I = 314                 # Replacement cost ($/kW)
            self.MO_I = 0                    # O&M cost ($/kw.year)

            # WT
            self.C_WT = 1200      # Capital cost ($) per KW
            self.R_WT = 1200      # Replacement Cost of WT Per KW
            self.MO_WT = 40      # O&M cost ($/year/kw)

            # Diesel generator
            self.C_DG = 818       # Capital cost ($/KW)
            self.R_DG = 818       # Replacement Cost ($/kW)
            self.MO_DG = 0.016    # O&M+ running cost ($/op.h)
            self.C_fuel = 1.281  # Fuel Cost ($/L)
            self.C_fuel_adj_rate = 2  # DG fuel cost yearly escalation  rate (if positive) and reduction rate (if negative)
            self.C_fuel_adj = self.C_fuel_adj_rate / 100

            # Battery
            self.C_B = 1450             # Capital cost ($/KWh)
            self.R_B = 1450             # Replacement Cost ($/kW)
            self.MO_B = 10                # Maintenance cost ($/kw.year)

            # Charger
            self.C_CH = 0 # Capital Cost ($)
            self.R_CH = 0  # Replacement Cost ($)
            self.MO_CH = 0  # O&M cost ($/year)

            # Heat Pump
            self.C_HP = 109.5  # Capital Cost ($/1000BTU/hr)
            self.R_HP = 109.5  # Replacement Cost ($/1000BTU/hr)
            self.MO_HP = 20  # O&M cost ($/year)

            # EV
            self.Cost_EV = 0
            self.R_EVB = 27000  # Replacement cost for EV battery ($)
            self.MO_EV = 0  # O&M cost ($/year)



        # Prices for Utility
        # Definition for the Utility Structures
        # 1 = flat rate
        # 2 = seasonal rate
        # 3 = monthly rate
        # 4 = tiered rate
        # 5 = seasonal tiered rate
        # 6 = monthly tiered rate
        # 7 = time of use rate
        # 8 = ultra-low pricing


        # Hourly Rate Structure
        self.rateStructure = 7

        # Fixed expenses
        self.Annual_expenses = 0 # Annual expenses in $ for grid if any

        self.Grid_sale_tax_rate = 0.986 # Sale tax percentage of grid electricity

        self.Grid_Tax = self.Grid_sale_tax_rate / 100

        self.Grid_Tax_amount = 0.0016 # Grid tax in $/kWh if any

        Grid_escalation_projection = 1

        if Grid_escalation_projection == 1:
            self.Grid_escalation_rate = np.full(self.n, 2) # Yearly escalation flat rate in grid electricity prices
        else:
            self.Grid_escalation_rate = np.array([5.7, 5.7, 5.7, 5.7, 5.7, 5.7, 5.7, 5.7, 5.7, 5.7, 5.7, 5.7, 5.7, 5.7, 5.7, 5.7, 5.7, 5.7, 5.7, 5.7, 5.7, 5.7, 5.7, 5.7, 5.7])  # Yearly escalation rate in grid electricity prices

        self.Grid_escalation = self.Grid_escalation_rate / 100

        self.Grid_credit = 58.23 * 2 # Credits offered by grid to users in $

        self.NEM_fee = 0 # Net metering one time setup fee

        # Monthly fixed charge structure
        self.Monthly_fixed_charge_system = 1

        if self.Monthly_fixed_charge_system == 1:  # Flat
            self.SC_flat = 15
            self.Service_charge = np.ones(12) * self.SC_flat
        else:  # Tiered
            self.SC_1 = 34.29  # tier 1 service charge
            self.Limit_SC_1 = 800  # limit for tier 1
            self.SC_2 = 46.54  # tier 2 service charge
            self.Limit_SC_2 = 1500  # limit for tier 2
            self.SC_3 = 66.29  # tier 3 service charge
            self.Limit_SC_3 = 1500  # limit for tier 3
            self.SC_4 = 66.29  # tier 4 service charge

            from service_charge import service_charge
            self.Service_charge = service_charge(self.daysInMonth, self.Eload_Previous, self.Limit_SC_1, self.SC_1, self.Limit_SC_2, self.SC_2, self.Limit_SC_3, self.SC_3, self.SC_4)

        # Hourly charges
        if self.rateStructure == 1:  # Flat rate
            self.flatPrice = 0.2

            from calcFlatRate import calcFlatRate
            self.Cbuy = calcFlatRate(self.flatPrice)

        elif self.rateStructure == 2:  # Seasonal rate
            self.seasonalPrices = np.array([0.0719, 0.0540])  # [summer, winter]
            self.season = np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0])  # define summer season 1= Summer

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
            self.seasonalTieredPrices = np.array([[0.075, 0.091, 0.091], [0.075, 0.091, 0.091]])  # [summer, winter][tier 1, 2, 3]
            self.seasonalTierMax = np.array([[600, 999999, 999999], [1000, 999999, 999999]])  # [summer, winter][tier 1, 2, 3] max kWh limits
            self.season = np.array([0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0])  # define summer season 1= Summer

            from calcSeasonalTieredRate import calcSeasonalTieredRate
            self.Cbuy = calcSeasonalTieredRate(self.seasonalTieredPrices, self.seasonalTierMax, self.Eload, self.season)

        elif self.rateStructure == 6:  # Monthly tiered rate
            self.monthlyTieredPrices = np.array([
                [0.404, 0.509, 0.509],
                [0.404, 0.509, 0.509],
                [0.404, 0.509, 0.509],
                [0.404, 0.509, 0.509],
                [0.404, 0.509, 0.509],
                [0.404, 0.509, 0.509],
                [0.404, 0.509, 0.509],
                [0.404, 0.509, 0.509],
                [0.404, 0.509, 0.509],
                [0.404, 0.509, 0.509],
                [0.404, 0.509, 0.509],
                [0.404, 0.509, 0.509]
            ])
            self.monthlyTierLimits = np.array([
                [343, 999999, 999999],
                [343, 999999, 999999],
                [343, 999999, 999999],
                [343, 999999, 999999],
                [343, 999999, 999999],
                [234, 999999, 999999],
                [234, 999999, 999999],
                [234, 999999, 999999],
                [234, 999999, 999999],
                [234, 999999, 999999],
                [343, 999999, 999999],
                [343, 999999, 999999]
            ])

            from calcMonthlyTieredRate import calcMonthlyTieredRate
            self.Cbuy = calcMonthlyTieredRate(self.monthlyTieredPrices, self.monthlyTierLimits, self.Eload)

        elif self.rateStructure == 7:  # Time of use rate
            self.onPrice = np.array([0.61, 0.38])  # prices for on-peak hours [summer, winter]
            self.midPrice = np.array([0.45, 0.36])   # prices for mid-peak hours [summer, winter]
            self.offPrice = np.array([0.4, 0.35])  # prices for off-peak hours [summer, winter]
            self.onHours = np.array([[16, 17, 18, 19, 20], [16, 17, 18, 19, 20]], dtype=object)  # on-peak hours [summer, winter]
            self.midHours = np.array([[15, 21, 22, 23], [15, 21, 22, 23]], dtype=object)  # mid-peak hours [summer, winter]
            self.season = np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0])  # define summer season 1= Summer
            self.treat_special_days_as_offpeak = False # True: Weekends and holidays will be priced as off-peak, False: Weekends and holidays will be treated like regular weekdays, and their pricing will follow peak/mid/off schedule.

            from calcTouRate import calcTouRate
            self.Cbuy = calcTouRate(self.year, self.onPrice, self.midPrice, self.offPrice, self.onHours, self.midHours, self.season, self.daysInMonth, self.holidays, self.treat_special_days_as_offpeak)

        elif self.rateStructure == 8:  # Ultra low Time of use rate
            # ULO pricing (convert from cents to dollars)
            self.onPrice = np.array([0.284, 0.284])  # On-peak: 28.4¢/kWh (same year-round)
            self.midPrice = np.array([0.122, 0.122])  # Mid-peak: 12.2¢/kWh (same year-round)
            self.offPrice = np.array([0.076, 0.076])  # Weekend off-peak: 7.6¢/kWh (same year-round)
            self.ultraLowPrice = np.array([0.028, 0.028])  # Ultra-low overnight: 2.8¢/kWh (same year-round)
            # Hour definitions (0-23)
            self.onHours = np.array([[16, 17, 18, 19, 20], [16, 17, 18, 19, 20]], dtype=object)  # 4 PM - 9 PM weekdays
            self.midHours = np.array([[7, 8, 9, 10, 11, 12, 13, 14, 15, 21, 22], [7, 8, 9, 10, 11, 12, 13, 14, 15, 21, 22]], dtype=object)  # 7 AM - 4 PM and 9 PM - 11 PM weekdays
            self.ultraLowHours = np.array([[23, 0, 1, 2, 3, 4, 5, 6], [23, 0, 1, 2, 3, 4, 5, 6]], dtype=object)  # 11 PM - 7 AM every day
            # Season definition (ULO is same year-round, but keeping structure)
            self.season = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])  # All winter pricing
            self.treat_special_days_as_offpeak = True
            from calcULTouRate import calcULTouRate
            self.Cbuy = calcULTouRate(self.year, self.onPrice, self.midPrice, self.offPrice, self.ultraLowPrice,
                           self.onHours, self.midHours, self.ultraLowHours, self.season, self.daysInMonth,
                           self.holidays, self.treat_special_days_as_offpeak)


        # Sell to the Grid
        self.sellStructure = 2

        if self.sellStructure == 1:
            self.Csell = np.full(8760, 0.1)

        elif self.sellStructure == 2:
            self.monthlysellprices = np.array([0.05799, 0.04829, 0.04621, 0.04256, 0.04030, 0.03991, 0.03963, 0.03976, 0.03781, 0.03656, 0.03615, 0.03461])

            from calcMonthlyRate import calcMonthlyRate
            self.Csell = calcMonthlyRate(self.monthlysellprices, self.daysInMonth)

        elif self.sellStructure == 3:
            self.Csell = self.Cbuy

        # Grid emission information
        # Emissions produced by Grid generators (kg/kWh)
        self.E_CO2 = 0
        self.E_SO2 = 0
        self.E_NOx = 0

        # Constraints for buying/selling from/to grid
        self.Pbuy_max = 50 # ceil(1.2 * max(self.Eload))  # kWh
        self.Psell_max = 50 # self.Pbuy_max


        # Definition for the Natural Gas Utility Structures
        NG_energycontent = 10.97 # Energy content of 1m3 Natual Gas in kWh
        Furnace_eff = 0.95 # Efficiency of heating furnace

        def cm2kwh(value, multiplier_factor = 1 / (NG_energycontent * Furnace_eff)):
            return value * multiplier_factor

        # 1 = flat rate
        # 2 = seasonal rate
        # 3 = monthly rate
        # 4 = tiered rate
        # 5 = seasonal tiered rate
        # 6 = monthly tiered rate
        # Prices for Utility


        # Hourly Rate Structure
        self.rateStructure_NG = 1

        # Fixed expenses
        self.Annual_expenses_NG = 0 # Annual expenses in $ for grid if any

        self.Grid_sale_tax_rate_NG = 13 # Sale tax percentage of grid electricity

        self.Grid_Tax_NG = self.Grid_sale_tax_rate_NG / 100

        self.Grid_Tax_amount_NG = cm2kwh(0.11) # Grid tax in $/m3 if any

        Grid_escalation_projection_NG = 1
        if Grid_escalation_projection_NG == 1:
            self.Grid_escalation_rate_NG = np.full(25, 2) # Yearly escalation flat rate in grid electricity prices
        else:
            self.Grid_escalation_rate_NG = np.array([5.7, 5.7, 5.7, 5.7, 5.7, 5.7, 5.7, 5.7, 5.7, 5.7, 5.7, 5.7, 5.7, 5.7, 5.7, 5.7, 5.7, 5.7, 5.7, 5.7, 5.7, 5.7, 5.7, 5.7, 5.7])  # Yearly escalation rate in grid electricity prices

        self.Grid_escalation_NG = self.Grid_escalation_rate / 100

        self.Grid_credit_NG = cm2kwh(0) # Credits offered by grid to users in $


        # Monthly fixed charge structure
        self.Monthly_fixed_charge_system_NG = 1

        if self.Monthly_fixed_charge_system_NG == 1:  # Flat
            self.SC_flat_NG = 18.59
            self.Service_charge_NG = np.ones(12) * self.SC_flat_NG

        # Hourly charges
        if self.rateStructure_NG == 1:  # Flat rate
            self.flatPrice_NG = cm2kwh(0.18)

            from calcFlatRate import calcFlatRate
            self.Cbuy_NG = calcFlatRate(self.flatPrice_NG)

        elif self.rateStructure_NG == 2:  # Seasonal rate
            self.seasonalPrices_NG = cm2kwh(np.array([0.0719, 0.0540]))  # [summer, winter]
            self.season_NG = np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0])  # define summer season 1= Summer

            from calcSeasonalRate import calcSeasonalRate
            self.Cbuy_NG = calcSeasonalRate(self.seasonalPrices_NG, self.season_NG, self.daysInMonth)

        elif self.rateStructure_NG == 3:  # Monthly rate
            self.monthlyPrices_NG = cm2kwh(np.array([0.54207, 0.53713, 0.38689, 0.30496, 0.28689, 0.28168, 0.30205, 0.28956, 0.26501, 0.26492, 0.3108, 0.40715]))

            from calcMonthlyRate import calcMonthlyRate
            self.Cbuy_NG = calcMonthlyRate(self.monthlyPrices_NG, self.daysInMonth)

        elif self.rateStructure_NG == 4:  # Tiered rate
            self.tieredPrices_NG = cm2kwh(np.array([0.1018, 0.1175, 0.1175]))  # prices for tiers
            self.tierMax_NG = np.array([300, 999999, 999999])  # max kWh limits for tiers

            from calcTieredRate import calcTieredRate
            self.Cbuy_NG = calcTieredRate(self.tieredPrices_NG, self.tierMax_NG, self.Tload, self.daysInMonth)

        elif self.rateStructure_NG == 5:  # Seasonal tiered rate
            self.seasonalTieredPrices_NG = cm2kwh(np.array([[0.075, 0.091, 0.091], [0.075, 0.091, 0.091]]))  # [summer, winter][tier 1, 2, 3]
            self.seasonalTierMax_NG = np.array([[600, 999999, 999999], [1000, 999999, 999999]])  # [summer, winter][tier 1, 2, 3] max kWh limits
            self.season_NG = np.array([0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0])  # define summer season 1= Summer

            from calcSeasonalTieredRate import calcSeasonalTieredRate
            self.Cbuy_NG = calcSeasonalTieredRate(self.seasonalTieredPrices_NG, self.seasonalTierMax_NG, self.Tload, self.season)

        elif self.rateStructure_NG == 6:  # Monthly tiered rate
            self.monthlyTieredPrices_NG = cm2kwh(np.array([
                [0.404, 0.509, 0.509],
                [0.404, 0.509, 0.509],
                [0.404, 0.509, 0.509],
                [0.404, 0.509, 0.509],
                [0.404, 0.509, 0.509],
                [0.404, 0.509, 0.509],
                [0.404, 0.509, 0.509],
                [0.404, 0.509, 0.509],
                [0.404, 0.509, 0.509],
                [0.404, 0.509, 0.509],
                [0.404, 0.509, 0.509],
                [0.404, 0.509, 0.509]
            ]))
            self.monthlyTierLimits_NG = np.array([
                [343, 999999, 999999],
                [343, 999999, 999999],
                [343, 999999, 999999],
                [343, 999999, 999999],
                [343, 999999, 999999],
                [234, 999999, 999999],
                [234, 999999, 999999],
                [234, 999999, 999999],
                [234, 999999, 999999],
                [234, 999999, 999999],
                [343, 999999, 999999],
                [343, 999999, 999999]
            ])

            from calcMonthlyTieredRate import calcMonthlyTieredRate
            self.Cbuy_NG = calcMonthlyTieredRate(self.monthlyTieredPrices_NG, self.monthlyTierLimits_NG, self.Tload)


        data = {'Eload': self.Eload, 'Eload_eh': self.Eload_eh, 'Eload_hp': self.Eload_hp, 'G': self.G, 'T': self.T, 'Vw': self.Vw, 'Cbuy': self.Cbuy, 'EV_P': self.EV_p}
        df = pd.DataFrame(data)
        df.to_csv('output/data/Inputs.csv', index=False)

InData = Input_Data()