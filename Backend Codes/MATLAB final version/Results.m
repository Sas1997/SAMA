Input_Data;

NT=length(Eload); % time step numbers

Npv=round(X(1));      % PV number
Nwt=round(X(2));      % WT number
Nbat=round(X(3));     % Battery pack number
N_DG=round(X(4));     % number of Diesel Generator
Cn_I=X(5);            % Inverter Capacity

Pn_PV=Npv*Ppv_r; % PV Total Capacity
Pn_WT=Nwt*Pwt_r; % WT Total Capacity
Cn_B=Nbat*Cbt_r; % Battery Total Capacity
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
    Cbw=R_B*Cn_B/(Cn_B*Q_lifetime*sqrt(ef_bat) );
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
I_Cost_without_incentives=C_PV*Pn_PV + C_WT*Pn_WT+ C_DG*Pn_DG+C_B*Cn_B+C_I*Cn_I +C_CH+Engineering_Costs*Pn_PV;
Total_incentives_received=I_Cost_without_incentives-I_Cost;

Top_DG=sum(Pdg>0)+1;
L_DG=TL_DG/Top_DG;
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
RC_DG(L_DG+1:L_DG:n)= R_DG*Pn_DG./(1+ir).^(L_DG+1:L_DG:n) ;
RC_B(L_B+1:L_B:n) = R_B*Cn_B./(1+ir).^(1.001*L_B:L_B:n) ;
RC_I(L_I+1:L_I:n) = R_I*Cn_I./(1+ir).^(1.001*L_I:L_I:n) ;
RC_CH(L_CH+1:L_CH:n) = R_CH./(1+ir).^(1.001*L_CH:L_CH:n) ;
R_Cost=RC_PV+RC_WT+RC_DG+RC_B+RC_I+RC_CH;

% Total M&O Cost ($/year)
MO_Cost=( MO_PV*Pn_PV + MO_WT*Pn_WT+ MO_DG*Pn_DG*sum(Pdg>0)+ MO_B*Cn_B+ MO_I*Cn_I +MO_CH)./(1+ir).^(1:n) ;

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

Grid_Cost= ((Annual_expenses+sum(Service_charge)+sum(Pbuy.*Cbuy)-sum(Psell.*Csell)).*1./(1+ir).^(1:n))*(1+Grid_Tax)*(Grid>0);
Grid_Cost_onlyG= ((Annual_expenses+sum(Service_charge)+sum(Eload.*Cbuy)).*1./(1+ir).^(1:n))*(1+Grid_Tax);
% Capital recovery factor
CRF=ir*(1+ir)^n/((1+ir)^n -1);

% Totall Cost
NPC=(((I_Cost+sum(R_Cost)+ sum(MO_Cost)+sum(C_Fu)-Salvage)*(1+System_Tax))+sum(Grid_Cost));
NPC_without_incentives=(((I_Cost_without_incentives+sum(R_Cost)+ sum(MO_Cost)+sum(C_Fu)-Salvage)*(1+System_Tax))+sum(Grid_Cost));
NPC_Grid=(sum(Grid_Cost_onlyG));
Operating_Cost=(CRF*(((sum(R_Cost)+ sum(MO_Cost)+sum(C_Fu)-Salvage)*(1+System_Tax))+sum(Grid_Cost)));

LCOE=CRF*NPC/sum(Eload-Ens+Psell);% Levelized Cost of Energy ($/kWh)
LCOE_without_incentives=CRF*NPC_without_incentives/sum(Eload-Ens+Psell);
LCOE_Grid=CRF*NPC_Grid/sum(Eload-Ens);
LEM=(DG_Emissions+Grid_Emissions)/sum(Eload-Ens); % Levelized Emissions(kg/kWh)

Ebmin=SOC_min*Cn_B;                                   % Battery minimum energy
Pb_min(2:NT)=(Eb(1:NT-1)-Ebmin)+Pdch(2:NT); % Battery minimum power in t=2,3,...,NT
Ptot=(Ppv+Pwt+Pb_min)*n_I+Pdg+Grid*Pbuy_max;          % total available power in system for each hour
DE=max((Eload-Ptot),0);                               % power shortage in each hour
LPSP=sum(Ens) /sum(Eload);

RE=1-sum(Pdg+Pbuy)/sum(Eload+Psell-Ens);%   sum(Ppv+Pwt-Edump)/sum(Eload+Psell-Ens);
RE(isnan(RE))=0;

Investment=zeros(1,n);
Investment(1)=I_Cost;
Salvage(n)=Salvage;
Salvage(1)=0;
Operating(1:n)=MO_PV*Pn_PV + MO_WT*Pn_WT+ MO_DG*Pn_DG+ MO_B*Cn_B+ MO_I*Cn_I+sum(Pbuy.*Cbuy)-sum(Psell.*Csell) ;
Fuel(1:n)=sum(C_fuel*q);

RC_PV(L_PV+1:L_PV:n)= R_PV*Pn_PV ;
RC_WT(L_WT+1:L_WT:n)= R_WT*Pn_WT ;
RC_DG(L_DG+1:L_DG:n)= R_DG*Pn_DG ;
RC_B(L_B+1:L_B:n) = R_B*Cn_B ;
RC_I(L_I+1:L_I:n) = R_I*Cn_I ;
Replacement=RC_PV+RC_WT+RC_DG+RC_B+RC_I;

Cash_Flow=[-Investment;-Operating; Salvage;-Fuel;-Replacement]';
figure()
bar(Cash_Flow,'stacked')
legend('Capital','Operating','Salvage','Fuel','Replacement')
title('Cash Flow')
xlabel('Year')
ylabel('$')

%%
disp( ' ')
disp( 'System Size ')
disp(['Cpv  (kW) = ' num2str(Pn_PV)])
if WT==1
disp(['Cwt  (kW) = ' num2str(Pn_WT)])
end
disp(['Cbat (kWh) = ' num2str(Cn_B)])
disp(['Cdg  (kW) = ' num2str(Pn_DG)])
disp(['Cinverter (kW) = ' num2str(Cn_I)])

disp(' ')
disp( 'Result: ')
disp(['NPC  = $' num2str(round(NPC,2))])
disp(['NPC without incentives = $' num2str(round(NPC_without_incentives,2))])
disp(['NPC for only Grid connected system = $' num2str(round(NPC_Grid,2))])
disp(['LCOE  = ' num2str(round(LCOE,2)) ' $/kWh '])
disp(['LCOE without incentives = ' num2str(round(LCOE_without_incentives,2)) ' $/kWh '])
disp(['LCOE for only Grid connected system = ' num2str(round(LCOE_Grid,2)) ' $/kWh '])
disp(['Operating Cost  = $' num2str(round(Operating_Cost,2))])
disp(['Initial Cost  = $' num2str(round(I_Cost,2))])
disp(['Initial Cost without incentives= $' num2str(round(I_Cost_without_incentives,2))])
disp(['Total incentives received= $' num2str(round(Total_incentives_received,2))])
disp(['RE  = ' num2str(100*RE) ' % '])
disp(['Total operation and maintenance cost  = $' num2str(sum(round(MO_Cost,2)))])
%%
disp(['PV Power  = ' num2str(sum(Ppv)) ' kWh '])
if WT==1
disp(['WT Power  = ' num2str(sum(Pwt)) ' kWh '])
end
disp(['DG Power  = ' num2str(sum(Pdg)) ' kWh '])
disp(['LPSP  = ' num2str(100*LPSP) ' % '])
disp(['Excess Electricity = ' num2str(sum(Edump)) ' kWh '])
%%%%%%%%%%%%%%%%%New edits
%%
if Grid==1
Total_Pbuy=(sum(Pbuy)*n)*(Grid>0);
Total_Psell=(sum(Psell)*n)*(Grid>0);
disp(['Total power bought from Grid= ' num2str(Total_Pbuy) ' kWh '])
disp(['Power sold to Grid= ' num2str(Total_Psell) ' kWh '])
disp(['Total Money paid to the Grid= $' num2str(round(sum(Grid_Cost),2))])
disp(['Grid Emissions   = ' num2str(Grid_Emissions) ' (kg/year) '])
end
%%
disp(['Total Money paid by the user= $' num2str(round(sum(NPC),2))])
%%
disp(['total fuel consumed by DG   = ' num2str(sum(q)) ' (Liter/year) '])
disp(['DG Emissions   = ' num2str(DG_Emissions) ' (kg/year) '])
disp(['LEM  = ' num2str(LEM) ' kg/kWh '])
%% Plot Results
if Grid==0
    Pbuy=0;
    Psell=0;
else

figure(2)
plot(Pbuy)
hold on
plot(Psell)
legend('Buy','sell')
ylabel('Pgrid (kWh)')
xlabel('t(hour)')

end

figure(3)
plot(Eload-Ens,'b-.','LineWidth',1)
hold on
plot(Pdg,'r')
plot(Pch-Pdch,'g')
plot(Ppv+Pwt,'--')
legend('Load-Ens','Pdg','Pbat','P_{RE}')

figure(4)
plot(Eb/Cn_B)
hold on
title('State of Charge')
ylabel('SOC')
xlabel('t[hour]')

%%%%Plot results for one specific day 
Day=180;
t1=Day*24+1;
t2=Day*24+24;

figure(5)
title(['Results for ' num2str(Day) ' -th day']) 
subplot(4,4,[1 5])
plot(Eload)
title('Load Profile')
ylabel('E_{load} [kWh]')
xlabel('t[hour]')
xlim([t1 t2])

subplot(4,4,2)
plot(G)
title('Plane of Array Irradiance')
ylabel('G[W/m^2]')
xlabel('t[hour]')
xlim([t1 t2])

subplot(4,4,6)
plot(T)
title('Ambient Temperature')
ylabel('T[^o C]')
xlabel('t[hour]')
xlim([t1 t2])

subplot(4,4,[3 8])
plot(Ppv)
title('PV Power')
ylabel('P_{pv} [kWh]')
xlabel('t[hour]')
xlim([t1 t2])

if WT==1

subplot(4,4,[3 4])
plot(Ppv)
title('PV Power')
ylabel('P_{pv} [kWh]')
xlabel('t[hour]')
xlim([t1 t2])

subplot(4,4,[7 8])
plot(Pwt)
title('WT Energy')
ylabel('P_{wt} [kWh]')
xlabel('t[hour]')
xlim([t1 t2])
end

subplot(4,4,[9 10])
plot(Pdg)
title('Diesel Generator Energy')
ylabel('E_{DG} [kWh]')
xlabel('t[hour]')
xlim([t1 t2])

subplot(4,4,11)
plot(Eb)
title('Battery Energy Level')
ylabel('E_{b} [kWh]')
xlabel('t[hour]')
xlim([t1 t2])

subplot(4,4,12)
plot(Eb/Cn_B)
title('State of Charge')
ylabel('SOC')
xlabel('t[hour]')
xlim([t1 t2])

subplot(4,4,13)
plot(Ens)
title('Loss of Power Suply')
ylabel('LPS[kWh]')
xlabel('t[hour]')
xlim([t1 t2])

subplot(4,4,14)
plot(Edump)
title('Dumped Energy')
ylabel('E_{dump} [kWh]')
xlabel('t[hour]')
xlim([t1 t2])

subplot(4,4,15)
bar(Pdch)
title('Battery decharge Energy')
ylabel('E_{dch} [kWh]')
xlabel('t[hour]')
xlim([t1 t2])

subplot(4,4,16)
bar(Pch)
title('Battery charge Energy')
ylabel('E_{ch} [kWh]')
xlabel('t[hour]')
xlim([t1 t2])


C_c = zeros(12,31);
index = 1;
for m=1:12
        index1 = index;
        for d=1:daysInMonth(m)
            Total_daily_load =sum(Eload(index1:index1+23));
            C_c(m,d) = Total_daily_load;
            index1 = index1 + 24;
        end
        index = (24 * daysInMonth(m)) + index;
end

EE_c=round(LCOE.*C_c,2);

figure(6)
EE_c(EE_c==0) = nan;
heatmap(EE_c, 'Colormap', jet);
ax = gca;
ax.YData =["January" "February" "March" "April" "May" "June" "July" "August" "September" "October" "November" "December"]

paperout=table([Ppv;Pdg;Pch;Pdch]);
writetable(paperout,'Output.xlsx')
