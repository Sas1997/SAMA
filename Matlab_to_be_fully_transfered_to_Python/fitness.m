
function Z=fitness(X)
global Eload Eload_Previous G T Vw
NT=length(Eload);  % time step numbers
Input_Data;

Npv=round(X(1));      % PV number
Nwt=round(X(2));      % WT number
Nbat=round(X(3));     % Battery pack number
N_DG=round(X(4));     % number of Diesel Generator
Cn_I=X(5);            % Inverter Capacity

Pn_PV=Npv*Ppv_r;  % PV Total Capacity
Pn_WT=Nwt*Pwt_r;  % WT Total Capacity
Cn_B=Nbat*Cbt_r;  % Battery Total Capacity
Pn_DG=N_DG*Cdg_r; % Diesel Total Capacity

%% PV Power Calculation
Tc=((T+(Tc_noct-Ta_noct)*(G/G_noct)*(1-(n_PV*(1-Tcof*Tref)/gama)))/(1+(Tc_noct-Ta_noct)*(G/G_noct)*((Tcof*n_PV)/gama))); % Module Temprature
Ppv = fpv*Pn_PV.*(G/Gref).*(1+Tcof.*(Tc-Tref)); % output power(kw)_hourly

%% Wind turbine Power Calculation
v1=Vw;     %hourly wind speed
v2=((h_hub/h0)^(alfa_wind_turbine))*v1; % v1 is the speed at a reference height;v2 is the speed at a hub height h2

Pwt=zeros(1,8760);
for t=1:1:8760
    
    if v2(t)<v_cut_in || v2(t)>v_cut_out 
        Pwt(t)=0;
    elseif v_cut_in<=v2(t)&& v2(t)<v_rated
        Pwt(t)=v2(t)^3 *(Pwt_r/(v_rated^3-v_cut_in^3))-(v_cut_in^3/(v_rated^3-v_cut_in^3))*(Pwt_r);
    elseif v_rated<=v2(t) && v2(t)<v_cut_out
        Pwt(t)=Pwt_r;
    else
        Pwt(t)=0;
    end
    Pwt(t)=Pwt(t)*Nwt;
end

%% Energy Management 
% Battery Wear Cost
if Cn_B>0
    Cbw=R_B*Cn_B/(Nbat*Q_lifetime*sqrt(ef_bat) );
else
    Cbw=0;
end

% DG Fix cost
cc_gen=b*Pn_DG*C_fuel+R_DG*Pn_DG/TL_DG+MO_DG;


[Eb,Pdg,Edump,Ens,Pch,Pdch,Pbuy,Psell]=...
    EMS(Ppv,Pwt,Eload,Cn_B,Nbat,Pn_DG,NT,SOC_max,SOC_min,SOC_initial,n_I,Grid,Cbuy,a,Cn_I,LR_DG,C_fuel,Pbuy_max,Psell_max,cc_gen,Cbw,self_discharge_rate,alfa_battery,c,k,Imax,Vnom,ef_bat);

q=(a*Pdg+b*Pn_DG).*(Pdg>0);   % Fuel consumption of a diesel generator 

%% installation and operation cost

% Total Investment cost ($)
I_Cost=C_PV*(1-RE_incentives)*Pn_PV + C_WT*(1-RE_incentives)*Pn_WT+ C_DG*Pn_DG+C_B*(1-RE_incentives)*Cn_B+C_I*(1-RE_incentives)*Cn_I +C_CH*(1-RE_incentives)+Engineering_Costs*(1-RE_incentives)*Pn_PV;

Top_DG=sum(Pdg>0)+1;
L_DG=round(TL_DG/Top_DG);
RT_DG=ceil(n/L_DG)-1; % Replecement time

% Total Replacement cost ($)
RC_PV= zeros(1,n);
RC_WT= zeros(1,n);
RC_DG= zeros(1,n);
RC_B = zeros(1,n);
RC_I = zeros(1,n);
RC_CH = zeros(1,n);

RC_PV(L_PV+1:L_PV:n)= R_PV*Pn_PV./(1+ir).^(1.001*L_PV:L_PV:n) ;
RC_WT(L_WT+1:L_WT:n)= R_WT*Pn_WT./(1+ir).^(1.001*L_WT:L_WT:n) ;
RC_DG(L_DG+1:L_DG:n)= R_DG*Pn_DG./(1+ir).^(1.001*L_DG:L_DG:n) ;
RC_B(L_B+1:L_B:n) = R_B*Cn_B./(1+ir).^(1.001*L_B:L_B:n) ;
RC_I(L_I+1:L_I:n) = R_I*Cn_I./(1+ir).^(1.001*L_I:L_I:n) ;
RC_CH(L_CH+1:L_CH:n) = R_CH./(1+ir).^(1.001*L_CH:L_CH:n) ;
R_Cost=RC_PV+RC_WT+RC_DG+RC_B+RC_I+RC_CH;

% Total M&O Cost ($/year)
MO_Cost=( MO_PV*Pn_PV + MO_WT*Pn_WT+ MO_DG*sum(Pn_DG>0)+ MO_B*Cn_B+ MO_I*Cn_I +MO_CH)./(1+ir).^(1:n) ;

% DG fuel Cost
C_Fu= sum(C_fuel*q)./(1+ir).^(1:n);

% Salvage
L_rem=(RT_PV+1)*L_PV-n; S_PV=(R_PV*Pn_PV)*L_rem/L_PV * 1/(1+ir)^n; % PV
L_rem=(RT_WT+1)*L_WT-n; S_WT=(R_WT*Pn_WT)*L_rem/L_WT * 1/(1+ir)^n; % WT
L_rem=(RT_DG+1)*L_DG-n; S_DG=(R_DG*Pn_DG)*L_rem/L_DG * 1/(1+ir)^n; % DG
L_rem=(RT_B +1)*L_B-n;  S_B =(R_B*Cn_B)*L_rem/L_B * 1/(1+ir)^n;
L_rem=(RT_I +1)*L_I-n;  S_I =(R_I*Cn_I)*L_rem/L_I * 1/(1+ir)^n;
L_rem=(RT_CH +1)*L_CH-n;  S_CH =(R_CH)*L_rem/L_CH * 1/(1+ir)^n;
Salvage=S_PV+S_WT+S_DG+S_B+S_I+S_CH;

% Emissions produced by Disesl generator (g)
DG_Emissions=sum( q.*(CO2 +NOx +SO2) )/1000; % total emissions (kg/year)
Grid_Emissions= (sum( Pbuy*(E_CO2+E_SO2+E_NOx) )/1000)*(Grid>0); % total emissions (kg/year)

Grid_Cost= ((Annual_expenses+((sum(Service_charge))*sum(Pbuy)>1)+sum(Pbuy.*Cbuy)-sum(Psell.*Csell)).*1./(1+ir).^(1:n))*(1+Grid_Tax)*(Grid>0);

% Capital recovery factor
CRF=ir*(1+ir)^n/((1+ir)^n -1);

% Totall Cost
NPC=(I_Cost+sum(R_Cost)+ sum(MO_Cost)+sum(C_Fu) -Salvage+sum(Grid_Cost))*(1+System_Tax);
Operating_Cost=(CRF*(sum(R_Cost)+ sum(MO_Cost)+sum(C_Fu) -Salvage+sum(Grid_Cost)))*(1+System_Tax);

if sum(Eload-Ens)>1
    LCOE=CRF*NPC/sum(Eload-Ens+Psell);                % Levelized Cost of Energy ($/kWh)
    LEM=(DG_Emissions+Grid_Emissions)/sum(Eload-Ens); % Levelized Emissions(kg/kWh)
else
    LCOE=100;
    LEM=100;
end

LPSP=sum(Ens)/sum(Eload);   

RE=1-sum(Pdg+Pbuy)/sum(Eload+Psell-Ens);
RE(isnan(RE))=0;

Z=LCOE+EM*LEM+10*(LPSP>LPSP_max)+10*(RE<RE_min)+100*(I_Cost>Budget)+... 
    100*max(0, LPSP-LPSP_max)+100*max(0, RE_min-RE)+100*max(0, I_Cost-Budget);

end