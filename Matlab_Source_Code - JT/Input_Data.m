
% Type of system (1: included, 0=not included)
PV=0;
WT=0;
DG=0;
Bat=0;
Grid=1;

EM=0; % 0: LCOE, 1:LCOE+LEM

Budget=200e3;   % Limit On Total Capital Cost

%%
n = 25;                  % life year of system (year)
n_ir=0.045;             % Nominal discount rate
e_ir=0.02;             % Expected inflation rate
ir=(n_ir-e_ir)/(1+e_ir); % real discount rate
%% JT
Tax_rate=5.8; %Percentage 
Tax=Tax_rate/100;

%%
LPSP_max=0.011; % Maximum loss of power supply probability
RE_min=0.75;    % minimum Renewable Energy
%%
Ppv_r=0.500;  % PV module rated power (kW)
Pwt_r=1;      % WT rated power (kW)
Cbt_r=1;      % Battery rated Capacity (kWh)
Cdg_r=0.5;    % Battery rated Capacity (kWh)

%% JT
%% Engineering Costs (Per/kW)
Installation_cost=305.57;
Overhead=549.87;
Sales_and_marketing=884.56;
Permiting_and_Inspection=205.95;
Electrical_BoS=399.11;
Structrual_BoS=89.90;
Supply_Chain_costs=357.22;
Profit_costs=287.72;

Engineering_Costs=(Profit_costs+Installation_cost+Overhead+Sales_and_marketing+Permiting_and_Inspection+Electrical_BoS+Structrual_BoS+Supply_Chain_costs)*(1+Tax);
%% Grid information 
% Emissions produced by Grid generators (g/kW)
E_CO2=1.43;
E_SO2=0.01;
E_NOx=0.39;
%% JT
%% PV data
% hourly_solar_radiation W
fpv=0.9;       % the PV derating factor [%]
Tcof=0;        % temperature coefficient
Tref=25;       % temperature at standard test condition
Tc_noct=46.5;      % Nominal operating cell temperature
Ta_noct=20;
G_noct=800;
gama=0.9;
n_PV=0.205;       % Efficiency of PV module
Gref = 1000 ;  % 1000 W/m^2

%% JT
C_PV = 314.18*(1+Tax) ;      % Capital cost ($) per KW
R_PV = 314.18*(1+Tax);       % Replacement Cost of PV modules Per KW
MO_PV = 29.49*(1+Tax) ;      % O&M  cost ($/year/kw)
L_PV=25;          % Life time (year)
% D_PV=0.01;        % PV yearly degradation
RT_PV=ceil(n/L_PV)-1;   % Replecement time

%% WT data
%% JT
h_hub=17;               % Hub height 
h0=43.6;                % anemometer height
nw=1;                   % Electrical Efficiency
v_cut_out=25;           % cut out speed
v_cut_in=2.5;           % cut in speed
v_rated=9.5;            % rated speed(m/s)
alfa_wind_turbine=0.14; % coefficient of friction ( 0.11 for extreme wind conditions, and 0.20 for normal wind conditions)

%% JT
C_WT = 1200*(1+Tax);      % Capital cost ($) per KW
R_WT = 1200*(1+Tax);      % Replacement Cost of WT Per KW
MO_WT = 40*(1+Tax) ;      % O&M  cost ($/year/kw)
L_WT=20;          % Life time (year)
n_WT=0.30;        % Efficiency of WT module
% D_WT=0.05;        % WT yearly degradation
RT_WT=ceil(n/L_WT)-1;   % Replecement time

%% JT
%% Diesel generator
C_DG = 308.64*(1+Tax);       % Capital cost ($/KWh)
R_DG = 308.64*(1+Tax);       % Replacement Cost ($/kW)
MO_DG = 0.064*(1+Tax);    % O&M+ running cost ($/op.h)
TL_DG=24000;      % Life time (h)
% n_DG=0.4;         % Efficiency
% D_DG=0.05;        % yearly degradation (%)
LR_DG=0.25;        % Minimum Load Ratio (%)

C_fuel=1.39*(1+Tax);  % Fuel Cost ($/L)
% Diesel Generator fuel curve
a=0.2730;          % L/hr/kW output
b=0.0330;          % L/hr/kW rated

% Emissions produced by Disesl generator for each fuel in littre [L]	g/L
CO2=2621.7;
CO = 16.34;
NOx = 6.6;
SO2 = 20;
%% JT
%% Battery data
C_B = 234.56*(1+Tax);              % Capital cost ($/KWh)
R_B = 234.56*(1+Tax);              % Repalacement Cost ($/kW)
MO_B=0*(1+Tax);                % Maintenance cost ($/kw.year)
L_B=15;                  % Life time (year)
SOC_min=0.2;
SOC_max=1;
SOC_initial=0.5;
% D_B=0.05;               % Degradation
RT_B=ceil(n/L_B)-1;     % Replecement time
Q_lifetime=8000;        % kWh
self_discharge_rate=0;  % Hourly self-discharge rate
alfa_battery=1;         % is the storage's maximum charge rate [A/Ah]
c=0.403;                % the storage capacity ratio [unitless] 
k=0.827;                % the storage rate constant [h-1]
Imax=16.7;              % the storage's maximum charge current [A]
Vnom=12;                % the storage's nominal voltage [V] 
ef_bat=0.8;             % storage DC-DC efficiency 
%% JT
%% Inverter
C_I = 229.38*(1+Tax);        % Capital cost ($/kW)
R_I = 229.38*(1+Tax);        % Replacement cost ($/kW)
MO_I =0*(1+Tax);         % O&M cost ($/kw.year)
L_I=25;           % Life time (year)
n_I=0.95;         % Efficiency
RT_I=ceil(n/L_I)-1; % Replecement time

%% JT
%% Charger
%%%%%%%%%%%%%%%New edits
if Bat==1
C_CH = 150*(1+Tax);  % Capital Cost ($)
R_CH = 150*(1+Tax);  % Replacement Cost ($)
MO_CH = 0*(1+Tax);   % O&M cost ($/year)
L_CH=25;     % Life time (year)
RT_CH=ceil(n/L_CH)-1; % Replecement time
else
C_CH = 0;  % Capital Cost ($)
R_CH = 0;  % Replacement Cost ($)
MO_CH = 0;   % O&M cost ($/year)
L_CH=25;     % Life time (year)
RT_CH=ceil(n/L_CH)-1; % Replecement time
end
%%%%%%%%%%%%%%%
%% JT
%% Prices for Utility

% 1 = flat rate
% 2 = seasonal rate
% 3 = monthly rate
% 4 = tiered rate
% 5 = seasonal tiered rate
% 6 = monthly tiered rate
% 7 = time of use rate
%% Define 
rateStructure = 6;
%%
Annual_expenses=0;
%%
% Months
months = ones(12,1);

% days in each month
daysInMonth = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31];
%%
% total monthly charge
if rateStructure == 1   % flat rate

    %Base Charge
    Base_charge = 12*ones(1,12);
    % price for flat rate
    flatPrice = 0.112;
    Cbuy = calcFlatRate(flatPrice);
    
elseif rateStructure == 2  % seasonal rate

    %Base Charge
    Base_charge = 12*ones(1,12);
    % prices for seasonal rate [summer, winter]
    seasonalPrices = {[0.17, 0.13]};
    % define summer season
    months(5:11) = 1;
    months(1:4) = 2;
    months(12)=2;
    Cbuy = calcSeasonalRate(seasonalPrices, months, daysInMonth);

elseif rateStructure == 3  % monthly rate

    %Base Charge
    Base_charge = 12*ones(1,12);
    % prices for monthly rate [Jan-Dec]
    monthlyPrices = {[0.15, 0.14, 0.13, 0.16, 0.11, 0.10, 0.12, 0.13, 0.14, 0.10, 0.15, 0.16]};
    Cbuy = calcMonthlyRate(monthlyPrices, daysInMonth);

elseif rateStructure == 4  % tiered rate

    %Base Charge
    Base_charge_tier_1=32;
    Base_charge_tier_2=43;  
    Base_charge_tier_3=67;
    % prices and max kwh limits [tier 1, 2, 3]
    tieredPrices = [0.1, 0.12, 0.15];
    tierMax = [680, 720, 1050];
[Cbuy , Base_charge] = calcTieredRate(Base_charge_tier_1,Base_charge_tier_2,Base_charge_tier_3,tieredPrices, tierMax, Eload, daysInMonth);

elseif rateStructure == 5  % seasonal tiered rate

    %Base Charge
    Base_charge_tier_1=32;
    Base_charge_tier_2=43;  
    Base_charge_tier_3=67;
    % prices and max kwh limits [summer,winter][tier 1, 2, 3]
    seasonalTieredPrices = {[0.05, 0.08, 0.14];
        [0.09, 0.13, 0.2]};
    seasonalTierMax = [[400, 800, 4000];
        [1000, 1500, 4000]];
    % define summer season
    months(5:11) = 1;
    months(1:4) = 2;
    months(12) = 2;
[Cbuy , Base_charge] = calcSeasonalTieredRate(Base_charge_tier_1,Base_charge_tier_2,Base_charge_tier_3,seasonalTieredPrices, seasonalTierMax, Eload, months);

elseif rateStructure == 6  % monthly tiered rate

    %Base Charge
    Base_charge_tier_1=2.30;
    Base_charge_tier_2=7.90;  
    Base_charge_tier_3=22.70;
    % prices and max kwh limits [Jan-Dec][tier 1, 2, 3]
    monthlyTieredPrices = [
        [0.19488, 0.25347, 0.25347];
        [0.19488, 0.25347, 0.25347];
        [0.19488, 0.25347, 0.25347];
        [0.19375, 0.25234, 0.25234];
        [0.19375, 0.25234, 0.25234];
        [0.19375, 0.25234, 0.33935];
        [0.18179, 0.24038, 0.32739];
        [0.18179, 0.24038, 0.32739];
        [0.18179, 0.24038, 0.32739];
        [0.19192, 0.25051, 0.25051];
        [0.19192, 0.25051, 0.25051];
        [0.19192, 0.25051, 0.25051]];
    
    monthlyTierLimits = [
        [500, 1500, 1501];
        [500, 1500, 1501];
        [500, 1500, 1501];
        [500, 1500, 1501];
        [500, 1500, 1501];
        [500, 1500, 1501];
        [500, 1500, 1501];
        [500, 1500, 1501];
        [500, 1500, 1501];
        [500, 1500, 1501];
        [500, 1500, 1501];
        [500, 1500, 1501]];
 [Cbuy , Base_charge] = calcMonthlyTieredRate(Base_charge_tier_1,Base_charge_tier_2,Base_charge_tier_3,monthlyTieredPrices, monthlyTierLimits, Eload);

elseif rateStructure == 7  % time of use rate

    %Base Charge
    Base_charge = 12*ones(1,12);
    % prices and time of use hours [summer,winter]
    onPrice = [0.1516, 0.3215];
    midPrice = [0.14, 0.1827];
    offPrice = [0.1098, 0.1323];
    onHours =[
        [17, 18, 19];
        [17, 18, 19]];
    midHours = {
        [12, 13, 14, 15, 16, 20, 21, 22, 23,24];
        [12, 13, 14, 15, 16, 20, 21, 22, 23,24]};
    offHours = {
        [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11];
        [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]};
    % define summer season
    months(5:10)=1;
    % Holidays definition based on the number of the day in 365 days format
    holidays = [10, 50, 76, 167, 298, 340];
    Cbuy = calcTouRate(onPrice, midPrice, offPrice, onHours, midHours, offHours, months, daysInMonth, holidays);

end

%%
Csell=0.1;
%%
Pbuy_max=ceil(1.2*max(Eload)); % kWh
Psell_max=Pbuy_max;