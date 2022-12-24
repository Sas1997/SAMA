
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
n_ir=0.0473;             % Nominal discount rate
e_ir=0.02;               % Expected inflation rate
ir=(n_ir-e_ir)/(1+e_ir); % real discount rate
Tax_rate=0; %Percentage 
Tax=Tax_rate/100;


LPSP_max=0.011; % Maximum loss of power supply probability
RE_min=0.75;    % minimum Renewable Energy

Ppv_r=0.500;  % PV module rated power (kW)
Pwt_r=1;      % WT rated power (kW)
Cbt_r=1;      % Battery rated Capacity (kWh)
Cdg_r=0.5;    % Battery rated Capacity (kWh)

%% Engineering Costs (Per/kW)
Installation_cost=0;
Overhead=0;
Sales_and_marketing=0;
Permiting_and_Inspection=0;
Electrical_BoS=0;
Structrual_BoS=0;
Supply_Chain_costs=0;

Engineering_Costs=(Installation_cost+Overhead+Sales_and_marketing+Permiting_and_Inspection+Electrical_BoS+Structrual_BoS+Supply_Chain_costs)*(1+Tax);
%% Grid information 
%Base Price
% Emissions produced by Grid generators (g/kW)
E_CO2=1.43;
E_SO2=0.01;
E_NOx=0.39;
%% PV data
% hourly_solar_radiation W
fpv=0.9;       % the PV derating factor [%]
Tcof=0;        % temperature coefficient
Tref=25;       % temperature at standard test condition
Tnoct=45;      % Nominal operating cell temperature
Gref = 1000 ;  % 1000 W/m^2


C_PV = 896*(1+Tax) ;      % Capital cost ($) per KW
R_PV = 896*(1+Tax);       % Replacement Cost of PV modules Per KW
MO_PV = 12*(1+Tax) ;      % O&M  cost ($/year/kw)
L_PV=25;          % Life time (year)
n_PV=0.205;       % Efficiency of PV module
D_PV=0.01;        % PV yearly degradation
RT_PV=ceil(n/L_PV)-1;   % Replecement time

%% WT data
h_hub=17;               % Hub height 
h0=43.6;                % anemometer height
nw=1;                   % Electrical Efficiency
v_cut_out=25;           % cut out speed
v_cut_in=2.5;           % cut in speed
v_rated=9.5;            % rated speed(m/s)
alfa_wind_turbine=0.14; % coefficient of friction ( 0.11 for extreme wind conditions, and 0.20 for normal wind conditions)


C_WT = 1200*(1+Tax);      % Capital cost ($) per KW
R_WT = 1200*(1+Tax);      % Replacement Cost of WT Per KW
MO_WT = 40*(1+Tax) ;      % O&M  cost ($/year/kw)
L_WT=20;          % Life time (year)
n_WT=0.30;        % Efficiency of WT module
D_WT=0.05;        % WT yearly degradation
RT_WT=ceil(n/L_WT)-1;   % Replecement time

%% Diesel generator
C_DG = 352*(1+Tax);       % Capital cost ($/KWh)
R_DG = 352*(1+Tax);       % Replacement Cost ($/kW)
MO_DG = 0.003*(1+Tax);    % O&M+ running cost ($/op.h)
TL_DG=131400;      % Life time (h)
n_DG=0.4;         % Efficiency
D_DG=0.05;        % yearly degradation (%)
LR_DG=0.25;        % Minimum Load Ratio (%)

C_fuel=1.24*(1+Tax);  % Fuel Cost ($/L)
% Diesel Generator fuel curve
a=0.2730;          % L/hr/kW output
b=0.0330;          % L/hr/kW rated

% Emissions produced by Disesl generator for each fuel in littre [L]	g/L
CO2=2621.7;
CO = 16.34;
NOx = 6.6;
SO2 = 20;

%% Battery data
C_B = 360*(1+Tax);              % Capital cost ($/KWh)
R_B = 360*(1+Tax);              % Repalacement Cost ($/kW)
MO_B=10*(1+Tax);                % Maintenance cost ($/kw.year)
L_B=5;                  % Life time (year)
SOC_min=0.2;
SOC_max=1;
SOC_initial=0.5;
D_B=0.05;               % Degradation
RT_B=ceil(n/L_B)-1;     % Replecement time
Q_lifetime=8000;        % kWh
self_discharge_rate=0;  % Hourly self-discharge rate
alfa_battery=1;         % is the storage's maximum charge rate [A/Ah]
c=0.403;                % the storage capacity ratio [unitless] 
k=0.827;                % the storage rate constant [h-1]
Imax=16.7;              % the storage's maximum charge current [A]
Vnom=12;                % the storage's nominal voltage [V] 
ef_bat=0.8;             % storage DC-DC efficiency 
%% Inverter
C_I = 788*(1+Tax);        % Capital cost ($/kW)
R_I = 788*(1+Tax);        % Replacement cost ($/kW)
MO_I =20*(1+Tax);         % O&M cost ($/kw.year)
L_I=25;           % Life time (year)
n_I=0.85;         % Efficiency
RT_I=ceil(n/L_I)-1; % Replecement time

%% Charger
%%%%%%%%%%%%%%%New edits
if Bat==1
C_CH = 150*(1+Tax);  % Capital Cost ($)
R_CH = 150*(1+Tax);  % Replacement Cost ($)
MO_CH = 5*(1+Tax);   % O&M cost ($/year)
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
%% Prices for Utility
% 1 = flat rate
% 2 = seasonal rate
% 3 = monthly rate
% 4 = tiered rate
% 5 = seasonal tiered rate
% 6 = monthly tiered rate
% 7 = time of use rate

rateStructure = 7;

% Months
months = ones(12,1);

% days in each month
daysInMonth = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31];

% total monthly charge
if rateStructure == 1   % flat rate
    % price for flat rate
    flatPrice = 0.112;
    Cbuy = calcFlatRate(flatPrice);
    
elseif rateStructure == 2  % seasonal rate
    % prices for seasonal rate [winter, summer]
    seasonalPrices = {[0.13, 0.17]};
    % define summer season
    months(4+1:11) = 2;
    Cbuy = calcSeasonalRate(seasonalPrices, months, daysInMonth);

elseif rateStructure == 3  % monthly rate
    % prices for monthly rate [Jan-Dec]
    monthlyPrices = {[0.15, 0.14, 0.13, 0.16, 0.11, 0.10, 0.12, 0.13, 0.14, 0.10, 0.15, 0.16]};
    Cbuy = calcMonthlyRate(monthlyPrices, daysInMonth);

elseif rateStructure == 4  % tiered rate
    % prices and max kwh limits [tier 1, 2, 3]
    tieredPrices = [0.1, 0.12, 0.15];
    tierMax = [500, 1000, 4000];
    Cbuy = calcTieredRate(tieredPrices, tierMax, Eload, daysInMonth);

elseif rateStructure == 5  % seasonal tiered rate
    % prices and max kwh limits [winter, summer][tier 1, 2, 3]
    seasonalTieredPrices = {[0.05, 0.08, 0.14];
        [0.09, 0.13, 0.2]};
    seasonalTierMax = [[400, 800, 4000];
        [1000, 1500, 4000]];
    % define summer season
    months(4+1:11) = 2;
    Cbuy = calcSeasonalTieredRate(seasonalTieredPrices, seasonalTierMax, Eload, months);

elseif rateStructure == 6  % monthly tiered rate
    % prices and max kwh limits [Jan-Dec][tier 1, 2, 3]
    monthlyTieredPrices = [
        [0.08, 0.10, 0.14];
        [0.08, 0.10, 0.14];
        [0.08, 0.10, 0.14];
        [0.08, 0.10, 0.14];
        [0.12, 0.14, 0.20];
        [0.12, 0.14, 0.20];
        [0.12, 0.14, 0.20];
        [0.12, 0.14, 0.20];
        [0.12, 0.14, 0.20];
        [0.12, 0.14, 0.20];
        [0.08, 0.10, 0.14];
        [0.08, 0.10, 0.14]];
    
    monthlyTierLimits = [
        [600, 1500, 4000];
        [600, 1500, 4000];
        [600, 1500, 4000];
        [600, 1500, 4000];
        [800, 1500, 4000];
        [800, 1500, 4000];
        [800, 1500, 4000];
        [800, 1500, 4000];
        [800, 1500, 4000];
        [800, 1500, 4000];
        [600, 1500, 4000];
        [600, 1500, 4000]];
    Cbuy = calcMonthlyTieredRate(monthlyTieredPrices, monthlyTierLimits, Eload);

elseif rateStructure == 7  % time of use rate
    % prices and time of use hours [winter, summer]
    onPrice = [0.1516, 0.3215];
    midPrice = [0, 0.1827];
    offPrice = [0.1098, 0.1323];
    onHours = [
        [17, 18, 19];
        [17, 18, 19]];
    midHours = {
        [];
        [12, 13, 14, 15, 16, 20, 21, 22, 23]};
    offHours = {
        [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 20, 21, 22, 23];
        [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]};
    % define summer season
    months(5+1:9) = 2;
    % Holidays definition based on the number of the day in 365 days format
    holidays = [10, 50, 76, 167, 298, 340];
    Cbuy = calcTouRate(onPrice, midPrice, offPrice, onHours, midHours, offHours, months, daysInMonth, holidays);

end
writematrix(Cbuy,'Cbuy.csv','Delimiter',',');
%%
Csell=0.1;
%%
Pbuy_max=ceil(1.2*max(Eload)); % kWh
Psell_max=Pbuy_max;