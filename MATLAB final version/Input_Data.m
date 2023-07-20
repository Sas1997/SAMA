%% Optimization
% PSO Parameters
MaxIt=10;      % Maximum Number of Iterations
nPop=5;        % Population Size (Swarm Size)
w=1;            % Inertia Weight
wdamp=0.99;     % Inertia Weight Damping Ratio
c1=2;           % Personal Learning Coefficient
c2=2;           % Global Learning Coefficient
% Multi-run
Run_Time=2; %Total number of runs in each click
%% Calendar
n = 25;                       % Life time of system in simulations (years)
year = 2023; % Specify the desired year
% Months
months = zeros(12,1);
% days in each month
% Determine the number of days in February based on whether it's a leap year
if isequal(year, floor(year/4)*4) % Leap year
    daysInMonth = [31, 29, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31];
else % Non-leap year
    daysInMonth = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31];
end
%% Reading global inputs
Data = csvread('Data.csv');
%% Electrcial load definitions 
%1=Hourly load based on the actual data
%2=Monthly average load
%3=Annual average load
%%
load_type=1;
%%
if load_type==1
Eload=Data(:,1)';
elseif load_type==2
    Monthly_average_load={[1, 2, 3, 4, 5, 6, 7, 8, 9, 10,11, 12]}; % Define the monthly averages for load here
    hCount = 1;
    for m=1:12
        for h=1:(24 * daysInMonth(m))
            Eload(hCount) = Monthly_average_load{1}(m);
            hCount = hCount +1;
        end
    end
else
Annual_average_load=1;  % Define the annual average for load here
Eload=Annual_average_load*ones(1,8760);
end

G=Data(:,2)';
T=Data(:,3)';
Vw=Data(:,4)';
Eload_Previous=Data(:,1)';
%% Other inputs
%% Technical data
%% Type of system (1: included, 0=not included)
PV=1;
WT=0;
DG=1;
Bat=1;
Grid=0;
%% Constranits 
LPSP_max_rate=0.00001; % Maximum loss of power supply probability
LPSP_max=LPSP_max_rate/100;

RE_min_rate=75;     % Minimum Renewable Energy Capacity
RE_min=RE_min_rate/100;

EM=0; % 0: LCOE, 1:LCOE+LEM
%% Rated capacity
Ppv_r=0.5;  % PV module rated power (kW)
Pwt_r=1;      % WT rated power (kW)
Cbt_r=1;      % Battery rated Capacity (kWh)
Cdg_r=0.5;    % DG rated Capacity (kW)
%% PV 
fpv=0.9;       % the PV derating factor [%]
Tcof=0;    % temperature coefficient [%/C]
Tref=25;       % temperature at standard test condition
Tc_noct=45;      % Nominal operating cell temperature
Ta_noct=20;
G_noct=800;
gama=0.9;
n_PV=0.2182;       % Efficiency of PV module
Gref = 1000 ;  % 1000 W/m^2
% D_PV=0.01;        % PV yearly degradation
L_PV=25;          % Life time (year)
RT_PV=ceil(n/L_PV)-1;   % Replecement time
%% Inverter
n_I=0.96;         % Efficiency
L_I=25;           % Life time (year)
RT_I=ceil(n/L_I)-1; % Replecement time
%% WT data
h_hub=17;               % Hub height 
h0=43.6;                % anemometer height
nw=1;                   % Electrical Efficiency
v_cut_out=25;           % cut out speed
v_cut_in=2.5;           % cut in speed
v_rated=9.5;            % rated speed(m/s)
alfa_wind_turbine=0.14; % coefficient of friction ( 0.11 for extreme wind conditions, and 0.20 for normal wind conditions)
% n_WT=0.30;        % Efficiency of WT module
% D_WT=0.05;        % WT yearly degradation
L_WT=20;          % Life time (year)
RT_WT=ceil(n/L_WT)-1;   % Replecement time
%% Diesel generator
% n_DG=0.4;         % Efficiency
% D_DG=0.05;        % yearly degradation (%)
LR_DG=0.25;        % Minimum Load Ratio (%)
% Diesel Generator fuel curve
a=0.2730;          % L/hr/kW output
b=0.0330;          % L/hr/kW rated
TL_DG=24000;      % Life time (h)
% Emissions produced by Disesl generator for each fuel in littre [L]	g/L
CO2=2621.7;
CO = 16.34;
NOx = 6.6;
SO2 = 20;
%% Battery
SOC_min=0.2;
SOC_max=1;
SOC_initial=0.5;
% D_B=0.05;               % Degradation
Q_lifetime=8000;        % kWh
self_discharge_rate=0;  % Hourly self-discharge rate
alfa_battery=1;         % is the storage's maximum charge rate [A/Ah]
c=0.403;                % the storage capacity ratio [unitless] 
k=0.827;                % the storage rate constant [h-1]
Imax=16.7;              % the storage's maximum charge current [A]
Vnom=12;                % the storage's nominal voltage [V] 
ef_bat=0.8;             % Round trip efficiency
L_B=7.5;                  % Life time (year)
RT_B=ceil(n/L_B)-1;     % Replecement time
%% Charger
L_CH=25;     % Life time (year)
RT_CH=ceil(n/L_CH)-1; % Replecement time


%% Economic Parameters 
n_ir_rate=4.5;             % Nominal discount rate
n_ir=n_ir_rate/100;           
e_ir_rate=2;                % Expected inflation rate
e_ir=e_ir_rate/100; 

ir=(n_ir-e_ir)/(1+e_ir); % real discount rate

Budget=200e3;   % Limit On Total Capital Cost 

Tax_rate=0;                 %Equipment sale tax Percentage 
System_Tax=Tax_rate/100;

RE_incentives_rate=30; %Federal tax credit percentage  
RE_incentives=RE_incentives_rate/100;
%% Pricing method 


Pricing_method=2; %1=Top down 2=bottom up

%If the pricing method is 1 i.e., Top-down method, then %System_tax% should be zero

%% Top-down price definition
if Pricing_method==1 
%% Pricing method 1/top down
Total_PV_price=2950;
%% NREL percentages 
r_PV=0.1812;
r_inverter=0.1492;
r_Installation_cost=0.0542;
r_Overhead=0.0881;
r_Sales_and_marketing=0.1356;
r_Permiting_and_Inspection=0.0712;
r_Electrical_BoS=0.1254;
r_Structrual_BoS=0.0542;
r_Profit_costs=0.1152;
r_Sales_tax=0.0271;
r_Supply_Chain_costs=0;
%% Engineering Costs (Per/kW)
Installation_cost=Total_PV_price*r_Installation_cost;
Overhead=Total_PV_price*r_Overhead;
Sales_and_marketing=Total_PV_price*r_Sales_and_marketing;
Permiting_and_Inspection=Total_PV_price*r_Permiting_and_Inspection;
Electrical_BoS=Total_PV_price*r_Electrical_BoS;
Structrual_BoS=Total_PV_price*r_Structrual_BoS;
Profit_costs=Total_PV_price*r_Profit_costs;
Sales_tax=Total_PV_price*r_Sales_tax;
Supply_Chain_costs=Total_PV_price*r_Supply_Chain_costs;
Engineering_Costs=(Sales_tax+Profit_costs+Installation_cost+Overhead+Sales_and_marketing+Permiting_and_Inspection+Electrical_BoS+Structrual_BoS+Supply_Chain_costs);
%% PV
C_PV = Total_PV_price*r_PV;         % Capital cost ($) per KW
R_PV = Total_PV_price*r_PV;        % Replacement Cost of PV modules Per KW
MO_PV = 28.12*(1+r_Sales_tax);      % PV O&M  cost ($/year/kw)
%% Inverter
C_I = Total_PV_price*r_inverter;         % Capital cost ($/kW)
R_I = Total_PV_price*r_inverter;         % Replacement cost ($/kW)
MO_I =3*(1+r_Sales_tax);         %Inverter O&M cost ($/kW.year)
%% WT
C_WT = 1200*(1+r_Sales_tax);      % Capital cost ($) per KW
R_WT = 1200*(1+r_Sales_tax);      % Replacement Cost of WT Per KW
MO_WT = 40*(1+r_Sales_tax);      % O&M  cost ($/year/kw)
%% Diesel generator
C_DG = 240.45*(1+r_Sales_tax);       % Capital cost ($/KW)
R_DG = 240.45*(1+r_Sales_tax);       % Replacement Cost ($/kW)
MO_DG = 0.064*(1+r_Sales_tax);     % O&M+ running cost ($/op.h)
C_fuel=1.39*(1+r_Sales_tax);             % Fuel Cost ($/L)
%% Battery
C_B = 458.06*(1+r_Sales_tax);              % Capital cost ($/kWh)
R_B = 458.06*(1+r_Sales_tax);              % Repalacement Cost ($/kW)
MO_B=10*(1+r_Sales_tax);                   % Maintenance cost ($/kWh.year)
%% Charger
if Bat==1
C_CH = 149.99*(1+r_Sales_tax);  % Capital Cost ($)
R_CH = 149.99*(1+r_Sales_tax);  % Replacement Cost ($)
MO_CH = 0*(1+r_Sales_tax);   % O&M cost ($/year)
else
C_CH = 0;  % Capital Cost ($)
R_CH = 0;  % Replacement Cost ($)
MO_CH = 0;   % O&M cost ($/year)
end
else %% Pricing method 2/bottom up
%% Engineering Costs (Per/kW)
Installation_cost=160;
Overhead=260;
Sales_and_marketing=400;
Permiting_and_Inspection=210;
Electrical_BoS=370;
Structrual_BoS=160;
Supply_Chain_costs=0;
Profit_costs=340;
Sales_tax=80;
Engineering_Costs=(Sales_tax+Profit_costs+Installation_cost+Overhead+Sales_and_marketing+Permiting_and_Inspection+Electrical_BoS+Structrual_BoS+Supply_Chain_costs);  

%% PV
C_PV = 2510;             % Capital cost ($) per KW
R_PV = 2510;             % Replacement Cost of PV modules Per KW
MO_PV = 28.12;      % O&M  cost ($/year/kw)
%% Inverter
C_I = 440;                 % Capital cost ($/kW)
R_I = 440;                  % Replacement cost ($/kW)
MO_I =3;                    % O&M cost ($/kw.year)
%% WT
C_WT = 1200;      % Capital cost ($) per KW
R_WT = 1200;      % Replacement Cost of WT Per KW
MO_WT = 40;      % O&M  cost ($/year/kw)
%% Diesel generator
C_DG = 240.45;       % Capital cost ($/KW)
R_DG = 240.45;       % Replacement Cost ($/kW)
MO_DG = 0.064;    % O&M+ running cost ($/op.h)
C_fuel=1.39;  % Fuel Cost ($/L)
%% Battery
C_B = 458.06;              % Capital cost ($/KWh)
R_B = 458.06;              % Repalacement Cost ($/kW)
MO_B=10;                % Maintenance cost ($/kw.year)
%% Charger
if Bat==1
C_CH = 149.99;  % Capital Cost ($)
R_CH = 149.99;  % Replacement Cost ($)
MO_CH = 0;   % O&M cost ($/year)
else
C_CH = 0;  % Capital Cost ($)
R_CH = 0;  % Replacement Cost ($)
MO_CH = 0;   % O&M cost ($/year)
end
end
%% Prices for Utility
%% Definition for the Utility Structures  
% 1 = flat rate
% 2 = seasonal rate
% 3 = monthly rate
% 4 = tiered rate
% 5 = seasonal tiered rate
% 6 = monthly tiered rate
% 7 = time of use rate
%% Grid emission information 
% Emissions produced by Grid generators (g/kW)
E_CO2=1.43;
E_SO2=0.01;
E_NOx=0.39;
%% Define the Utility structure for analysis 
rateStructure = 7;
%% Fixed expenses 
Annual_expenses=0;
Grid_sale_tax_rate=9.5;
Grid_Tax=Grid_sale_tax_rate/100;
%% Monthly fixed charge
Monthly_fixed_charge_system=2;        %1:flat %2:tier based
%% Service Charge (SC)
 if Monthly_fixed_charge_system==1
 %% Flat
 SC_flat=12;
 Service_charge = SC_flat*ones(1,12);
 else
 %% Tiered
 SC_1=2.30;         %tier 1 service charge
 Limit_SC_1=500;    %limit for tier 1 
 SC_2=7.9;          %tier 2 service charge
 Limit_SC_2=1500;   %limit for tier 2
 SC_3=22.7;         %tier 3 service charge
 Limit_SC_3=1500;   %limit for tier 3
 SC_4=22.7;         %tier 4 service charge
    hourCount = 1;
    for m=1:12
        monthlyLoad = 0;
        for h=1:(24 * daysInMonth(m))
            monthlyLoad= monthlyLoad + Eload_Previous(hourCount);
        hourCount = hourCount + 1;
        end
        totalmonthlyload(m,1)=monthlyLoad;
    end
    if max(totalmonthlyload(m,1))<Limit_SC_1
        Service_charge = SC_1*ones(1,12);
    elseif max(totalmonthlyload(m,1))<Limit_SC_2
        Service_charge = SC_2*ones(1,12);
    elseif max(totalmonthlyload(m,1))<Limit_SC_3
        Service_charge = SC_3*ones(1,12);
    else
        Service_charge = SC_4*ones(1,12);
    end
 end
%% Hourly charges
if rateStructure == 1   % flat rate

    % price for flat rate
    flatPrice = 0.112;
    Cbuy = calcFlatRate(flatPrice);
    
elseif rateStructure == 2  % seasonal rate

    % prices for seasonal rate [summer, winter]
    seasonalPrices = {[0.17, 0.13]};
    % define summer season
    months(5:10)=2;
    months(1:4)=1;
    months(11:12)=1;
    Cbuy = calcSeasonalRate(seasonalPrices, months, daysInMonth);

elseif rateStructure == 3  % monthly rate

    % prices for monthly rate [Jan-Dec]
    monthlyPrices = {[0.15, 0.14, 0.13, 0.16, 0.11, 0.10, 0.12, 0.13, 0.14, 0.10, 0.15, 0.16]};
    Cbuy = calcMonthlyRate(monthlyPrices, daysInMonth);

elseif rateStructure == 4  % tiered rate

    % prices and max kwh limits [tier 1, 2, 3]
    tieredPrices = [0.1, 0.12, 0.15];
    tierMax = [680, 720, 1050];
[Cbuy] = calcTieredRate(tieredPrices, tierMax, Eload, daysInMonth);

elseif rateStructure == 5  % seasonal tiered rate

    % prices and max kwh limits [summer,winter][tier 1, 2, 3]
    seasonalTieredPrices = {[0.05, 0.08, 0.14];
        [0.09, 0.13, 0.2]};
    seasonalTierMax = [[400, 800, 4000];
        [1000, 1500, 4000]];
    % define summer season
    months(5:10)=2;
    months(1:4)=1;
    months(11:12)=1;

[Cbuy] = calcSeasonalTieredRate(seasonalTieredPrices, seasonalTierMax, Eload, months);

elseif rateStructure == 6  % monthly tiered rate

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
 [Cbuy] = calcMonthlyTieredRate(monthlyTieredPrices, monthlyTierLimits, Eload);

elseif rateStructure == 7  % time of use rate

    % prices and time of use hours [summer,winter]
    onPrice = [0.17, 0.17];
    midPrice = [0.113, 0.113];
    offPrice = [0.083, 0.083];
    onHours =[
        [11, 12, 13,14,15,16];
        [7,8,9,10,17,18]];
    midHours = {
        [7,8,9,10,17,18];
        [11, 12, 13,14,15,16]};
    offHours = {
        [1, 2, 3, 4, 5, 6, 19,20,21,22,23,24];
        [1, 2, 3, 4, 5, 6, 19,20,21,22,23,24]};
    % define summer season
    months(5:10)=1;
    % Holidays definition based on the number of the day in 365 days format
    holidays = [10, 50, 76, 167, 298, 340];
    Cbuy = calcTouRate(year,onPrice, midPrice, offPrice, onHours, midHours, offHours, months, daysInMonth, holidays);

end

%% Sell electricity to the grid
Csell=0.14; %Cost of selling electricity to the grid per kWh
%% Constraints for selling to grid
Pbuy_max=6; %ceil(1.2*max(Eload)); % kWh
Psell_max=200;%Pbuy_max;