import numpy as np

from simplified_rate_structures import calcTouCbuy

class Data():
    def __init__(self, Eload, G, T, Vw):
        self.PV = 1 # Type of system (1: included, 0=not included)
        self.WT = 0 # Type of system (1: included, 0=not included)
        self.DG = 1 # Type of system (1: included, 0=not included)
        self.Bat = 1 # Type of system (1: included, 0=not included)
        self.Grid = 0 
        self.EM = 0 # 0: LCOE, 1:LCOE+LEM
        self.Budget = 200e3 # Limit On Total Capital Cost
        self.n = 25  # life year of system (year)
        self.n_ir=0.0473 # Nominal discount rate
        self.e_ir=0.02 # Expected inflation rate
        self.ir=(self.n_ir-self.e_ir)/(1+self.e_ir) # real discount rate
        self.LPSP_max=0.011 # Maximum loss of power supply probability
        self.RE_min=0.75    # minimum Renewable Energy
        self.Ppv_r=0.500  # PV module rated power (kW)
        self.Pwt_r=1      # WT rated power (kW)
        self.Cbt_r=1      # Battery rated Capacity (kWh)
        self.Cdg_r=0.5    # Battery rated Capacity (kWh)

        # PV data
        # hourly_solar_radiation W
        self.fpv=0.9       # the PV derating factor [%]
        self.Tcof=0        # temperature coefficient
        self.Tref=25       # temperature at standard test condition
        self.Tnoct=45      # Nominal operating cell temperature
        self.Gref = 1000   # 1000 W/m^2

        self.C_PV = 896       # Capital cost ($) per KW
        self.R_PV = 896       # Replacement Cost of PV modules Per KW
        self.MO_PV = 12       # O&M  cost ($/year/kw)
        self.L_PV=25          # Life time (year)
        self.n_PV=0.205       # Efficiency of PV module
        self.D_PV=0.01        # PV yearly degradation
        self.CE_PV=50         # Engineering cost of system per kW for first year
        self.RT_PV=np.ceil(self.n/self.L_PV)-1   # Replecement time

        # WT data
        self.h_hub=17               # Hub height 
        self.h0=43.6                # anemometer height
        self.nw=1                   # Efficiency
        self.v_cut_out=25           # cut out speed
        self.v_cut_in=2.5           # cut in speed
        self.v_rated=9.5            # rated speed(m/s)
        self.alfa_wind_turbine=0.14 # coefficient of friction ( 0.11 for extreme wind conditions, and 0.20 for normal wind conditions)

        self.C_WT = 1200      # Capital cost ($) per KW
        self.R_WT = 1200      # Replacement Cost of WT Per KW
        self.MO_WT = 40       # O&M  cost ($/year/kw)
        self.L_WT=20          # Life time (year)
        self.n_WT=0.30        # Efficiency of WT module
        self.D_WT=0.05        # PV yearly degradation
        self.RT_WT=np.ceil(self.n/self.L_WT)-1   # Replecement time

        # Diesel generator
        self.C_DG = 352       # Capital cost ($/KWh)
        self.R_DG = 352       # Replacement Cost ($/kW)
        self.MO_DG = 0.003    # O&M+ running cost ($/op.h)
        self.TL_DG=131400     # Life time (h)
        self.n_DG=0.4         # Efficiency
        self.D_DG=0.05        # yearly degradation (%)
        self.LR_DG=0.25       # Minimum Load Ratio (%)

        self.C_fuel=1.24  # Fuel Cost ($/L)
        # Diesel Generator fuel curve
        self.a=0.2730          # L/hr/kW output
        self.b=0.0330          # L/hr/kW rated

        # Emissions produced by Disesl generator for each fuel in littre [L]	g/L
        self.CO2=2621.7
        self.CO = 16.34
        self.NOx = 6.6
        self.SO2 = 20

        # Battery data
        self.C_B = 360              # Capital cost ($/KWh)
        self.R_B = 360              # Repalacement Cost ($/kW)
        self.MO_B=10                # Maintenance cost ($/kw.year)
        self.L_B=5                  # Life time (year)
        self.SOC_min=0.2
        self.SOC_max=1
        self.SOC_initial=0.5
        self.D_B=0.05               # Degradation
        self.RT_B=np.ceil(self.n/self.L_B)-1     # Replecement time
        self.Q_lifetime=8000        # kWh
        self.self_discharge_rate=0  # Hourly self-discharge rate
        self.alfa_battery=1         # is the storage's maximum charge rate [A/Ah]
        self.c=0.403                # the storage capacity ratio [unitless] 
        self.k=0.827                # the storage rate constant [h-1]
        self.Imax=16.7              # the storage's maximum charge current [A]
        self.Vnom=12                # the storage's nominal voltage [V] 
        self.ef_bat=0.8             # storage DC-DC efficiency 
        # Inverter
        self.C_I = 788        # Capital cost ($/kW)
        self.R_I = 788        # Replacement cost ($/kW)
        self.MO_I =20         # O&M cost ($/kw.year)
        self.L_I=25           # Life time (year)
        self.n_I=0.85         # Efficiency
        self.RT_I=np.ceil(self.n/self.L_I)-1 # Replecement time

        # Charger
        self.C_CH = 150  # Capital Cost ($)
        self.R_CH = 150  # Replacement Cost ($)
        self.MO_CH = 5   # O&M cost ($/year)
        self.L_CH=25     # Life time (year)
        self.RT_CH=np.ceil(self.n/self.L_CH)-1 # Replacement time

        # self.Cbuy = getTotalCharge()
        onPrice = [0.1516, 0.3215]
        midPrice = [0, 0.1827]
        offPrice = [0.1098, 0.1323]
        onHours = [
            [17, 18, 19],
            [17, 18, 19]
        ]
        midHours = [
            [],
            [12, 13, 14, 15, 16, 20, 21, 22, 23]
        ]
        offHours = [
            [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 20, 21, 22, 23],
            [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
        ]
        # define summer season
        months = np.zeros(12)
        months[5:9] = 1
        # days in each month
        daysInMonth = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
        # Holidays definition based on the number of the day in 365 days format
        holidays = [10, 50, 76, 167, 298, 340]

        self.Cbuy = calcTouCbuy(onPrice, midPrice, offPrice, onHours, midHours, offHours, months, daysInMonth, holidays)
        self.Csell = 0.1
        
        self.Pbuy_max=np.ceil(1.2*max(Eload)) # kWh
        self.Psell_max=self.Pbuy_max

        # Emissions produced by Grid generators (g/kW)
        self.E_CO2=1.43
        self.E_SO2=0.01
        self.E_NOx=0.39
    
    def set_user_data(self, **kwargs):
        self.__dict__.update(kwargs)
