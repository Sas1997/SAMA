import numpy as np
import pandas as pd
#%%
path='Data.csv'
Data = pd.read_csv(path, header=None).values
Eload = Data[:,0]
G = Data[:,1]
T = Data[:,2]
Vw = Data[:,3]

#%%
def calcTouCbuy(Day,Month,holidays):
   P_peak=0.17;
   P_mid=0.113;
   P_offpeak=0.083;
   Cbuy=np.zeros(8760)
   # Winter
   Tp_w=[7,8,9,10,17,18];
   Tm_w=list(range(11,17));
   Toff_w=[1,2,3,4,5,6,19,20,21,22,23,24];
   #Summer
   Tp_s=[11,12,13,14,15,16];
   Tm_s=[7,8,9,10, 17,18];
   Toff_s=[1,2,3,4,5,6, 19,20,21,22,23,24];

   for m in range(1,13):
       t_start=(24*np.sum(Day[0:m-1])+1).astype(int)
       t_end=(24*np.sum(Day[0:m])).astype(int)
       t_index=list(range(t_start,t_end+1))
       t_index=np.array(t_index).astype(int)
       if Month[m-1]==1:     #for summer
           tp=np.array(Tp_s);
           tm=np.array(Tm_s);
           toff=np.array(Toff_s);
       else:               #for winter
            tp=np.array(Tp_w);
            tm=np.array(Tm_w)
            toff=np.array(Toff_w);
        
       Cbuy[t_index-1]=P_offpeak
       for d in range(1,Day[m-1]+1):
           idx0=np.array(t_index[tp]+24*(d-1))-1
           Cbuy[idx0-1]=P_peak;
           idx1=np.array(t_index[tm]+24*(d-1))-1
           Cbuy[idx1-1]=P_mid;
           
       # for d in range(1,365):
       #     if  d in holidays:
       #         st=24*(d-1)+1
       #         ed=24*d
       #         Cbuy[range(st,ed)]=P_offpeak  
   return Cbuy

#Type of system (1: included, 0=not included)
PV=1
WT=1
DG=1
Bat=1
Grid=1
EM=0  # 0: LCOE, 1:LCOE+LEM
Budget=200e3    # Limit On Total Capital Cost

# %%
n = 25;                  # life year of system (year)
n_ir=0.0473;             # Nominal discount rate
e_ir=0.02;               # Expected inflation rate
ir=(n_ir-e_ir)/(1+e_ir); # real discount rate
LPSP_max=0.011; # Maximum loss of power supply probability
RE_min=0.75;    # minimum Renewable Energy
Ppv_r=0.500;  # PV module rated power (kW)
Pwt_r=1;      # WT rated power (kW)
Cbt_r=1;      # Battery rated Capacity (kWh)
Cdg_r=0.5;   # Battery rated Capacity (kWh)
#%% PV data
# hourly_solar_radiation W
fpv=0.9;       # the PV derating factor [%]
Tcof=0;        # temperature coefficient
Tref=25;       # temperature at standard test condition
Tnoct=45;      # Nominal operating cell temperature
Gref = 1000 ;  # 1000 W/m^2
C_PV = 896 ;      # Capital cost ($) per KW
R_PV = 896;       # Replacement Cost of PV modules Per KW
MO_PV = 12 ;      # O&M  cost ($/year/kw)
L_PV=25;          # Life time (year)
n_PV=0.205;       # Efficiency of PV module
D_PV=0.01;        # PV yearly degradation
CE_PV=50;         # Engineering cost of system per kW for first year
RT_PV=np.ceil(n/L_PV)-1;   # Replecement time

#%% WT data
h_hub=17;               # Hub height 
h0=43.6;                # anemometer height
nw=1;                   # Electrical Efficiency
v_cut_out=25;           # cut out speed
v_cut_in=2.5;           # cut in speed
v_rated=9.5;            # rated speed(m/s)
alfa_wind_turbine=0.14; # coefficient of friction ( 0.11 for extreme wind conditions, and 0.20 for normal wind conditions)
C_WT = 1200;      # Capital cost ($) per KW
R_WT = 1200;      # Replacement Cost of WT Per KW
MO_WT = 40 ;      # O&M  cost ($/year/kw)
L_WT=20;          # Life time (year)
n_WT=0.30;        # Efficiency of WT module
D_WT=0.05;        # WT yearly degradation
RT_WT=np.ceil(n/L_WT)-1;   # Replecement time

#%% Diesel generator
C_DG = 352;       # Capital cost ($/KWh)
R_DG = 352;       # Replacement Cost ($/kW)
MO_DG = 0.003;    # O&M+ running cost ($/op.h)
TL_DG=131400;      # Life time (h)
n_DG=0.4;         # Efficiency
D_DG=0.05;        # yearly degradation (%)
LR_DG=0.25;        # Minimum Load Ratio (%)
C_fuel=1.24;  # Fuel Cost ($/L)
## Diesel Generator fuel curve
a=0.2730;          # L/hr/kW output
b=0.0330;          # L/hr/kW rated
#% Emissions produced by Disesl generator for each fuel in littre [L]	g/L
CO2=2621.7;
CO = 16.34;
NOx = 6.6;
SO2 = 20;

#%% Battery data
C_B = 360;              # Capital cost ($/KWh)
R_B = 360;              # Repalacement Cost ($/kW)
MO_B=10;                # Maintenance cost ($/kw.year)
L_B=5;                  # Life time (year)
SOC_min=0.2;
SOC_max=1;
SOC_initial=0.5;
D_B=0.05;               # Degradation
RT_B=np.ceil(n/L_B)-1;     # Replecement time
Q_lifetime=8000;        # kWh
self_discharge_rate=0;  # Hourly self-discharge rate
alfa_battery=1;         # is the storage's maximum charge rate [A/Ah]
c=0.403;                # the storage capacity ratio [unitless] 
k=0.827;                # the storage rate constant [h-1]
Imax=16.7;              # the storage's maximum charge current [A]
Vnom=12;                # the storage's nominal voltage [V] 
ef_bat=0.8;             # storage DC-DC efficiency 
#%% Inverter
C_I = 788;        # Capital cost ($/kW)
R_I = 788;        # Replacement cost ($/kW)
MO_I =20;         # O&M cost ($/kw.year)
L_I=25;           # Life time (year)
n_I=0.85;         # Efficiency
RT_I=np.ceil(n/L_I)-1; # Replecement time

# %% Charger
#New edits
if Bat==1:
    C_CH = 150;  # Capital Cost ($)
    R_CH = 150;  # Replacement Cost ($)
    MO_CH = 5;   # O&M cost ($/year)
    L_CH=25;     # Life time (year)
    RT_CH=np.ceil(n/L_CH)-1; # Replecement time
else:
    C_CH = 0;  # Capital Cost ($)
    R_CH = 0;  # Replacement Cost ($)
    MO_CH = 0;   # O&M cost ($/year)
    L_CH=25;     # Life time (year)
    RT_CH=np.ceil(n/L_CH)-1; # Replecement time

#%% Price
# Winter
onPrice = [0.1516, 0.3215]
midPrice = [0, 0.1827]
offPrice = [0.1098, 0.1323]
onHours = [[17, 18, 19],[17, 18, 19]]
midHours = [[],
    [12, 13, 14, 15, 16, 20, 21, 22, 23]]
offHours = [[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 20, 21, 22, 23],
    [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]]
# define summer season
months = np.zeros(13)
months[4:10]=1; # Summer
# days in each month
daysInMonth = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
# Holidays definition based on the number of the day in 365 days format
holidays = [10, 50, 76, 167, 298, 340]

Cbuy = calcTouCbuy(daysInMonth,months, holidays)
Csell = 0.1

Pbuy_max=np.ceil(1.2*max(Eload)) # kWh
Psell_max=Pbuy_max

## Emissions produced by Grid generators (g/kW)
E_CO2=1.43;
E_SO2=0.01;
E_NOx=0.39;
