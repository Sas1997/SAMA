
import numpy as np
from Input_Data import *
from EMS import energy_management


def fitness(X, Eload, G, T, Vw):
    
    # if(len(X))==1:
    #     X=X[0]
    

    # reshape arrays
    nVar = len(X)
    Eload_vector = np.tile(Eload, (nVar, 1))
    G_vector = np.tile(G, (nVar, 1))
    T_vector = np.tile(T, (nVar, 1))
    Vw_vector = np.tile(Vw, (nVar, 1))

    NT=len(Eload);              # time step numbers
    Npv=np.around(X[:,0]);      # PV numbers
    Nwt=np.around(X[:,1]);      # WT numbers
    Nbat=np.around(X[:,2]);     # Battery pack numbers
    N_DG=np.around(X[:,3]);     # number of Diesel Generator
    Cn_I=X[:,4];                   # Inverter Capacity

    shape = np.swapaxes(Eload_vector, Eload_vector.ndim-1, 0).shape
    Npv = np.swapaxes(np.broadcast_to(Npv, shape), Eload_vector.ndim-1, 0)
    Nwt = np.swapaxes(np.broadcast_to(Nwt, shape), Eload_vector.ndim-1, 0)
    Nbat = np.swapaxes(np.broadcast_to(Nbat, shape), Eload_vector.ndim-1, 0)
    N_DG = np.swapaxes(np.broadcast_to(N_DG, shape), Eload_vector.ndim-1, 0)
    Cn_I = np.swapaxes(np.broadcast_to(Cn_I, shape), Eload_vector.ndim-1, 0)

    Pn_PV=Npv*Ppv_r;   # PV Total Capacity
    Pn_WT=Nwt*Pwt_r;   # WT Total Capacity
    Cn_B=Nbat*Cbt_r;   # Battery Total Capacity
    Pn_DG=N_DG*Cdg_r;  # Diesel Total Capacity

    # Pn_PV = np.swapaxes(np.broadcast_to(Pn_PV, shape), Eload_vector.ndim-1, 0)
    # Pn_WT = np.swapaxes(np.broadcast_to(Pn_WT, shape), Eload_vector.ndim-1, 0)
    # Cn_B = np.swapaxes(np.broadcast_to(Cn_B, shape), Eload_vector.ndim-1, 0)
    # Pn_DG = np.swapaxes(np.broadcast_to(Pn_DG, shape), Eload_vector.ndim-1, 0)
    
    #%% PV Power Calculation
    Tc   = T+(((Tnoct-20)/800)*G); # Module Temprature
    Ppv = fpv*Pn_PV*(G_vector/Gref)*(1+Tcof*(Tc-Tref)); # output power(kw)_hourly
    
    # %% Wind turbine Power Calculation
    v1=Vw_vector;     #hourly wind speed
    v2=((h_hub/h0)**(alfa_wind_turbine))*v1 # v1 is the speed at a reference height;v2 is the speed at a hub height h2
    Pwt=np.zeros((nVar,8760))


    Pwt[v2<v_cut_in]=0
    Pwt[v2>v_cut_out]=0
    true_value=np.logical_and(v_cut_in<=v2,v2<v_rated)
    Pwt[np.logical_and(v_cut_in<=v2,v2<v_rated)]=v2[true_value]**3 *(Pwt_r/(v_rated**3-v_cut_in**3))-(v_cut_in**3/(v_rated**3-v_cut_in**3))*(Pwt_r);
    Pwt[np.logical_and(v_rated<=v2,v2<v_cut_out)]=Pwt_r
    Pwt=Pwt*Nwt;
    
    
    #%% Energy Management 
    #% Battery Wear Cost
    Cbw = np.where(Cn_B>0, R_B*Cn_B/(Nbat*Q_lifetime*np.sqrt(ef_bat)), 0)
    # if Cn_B>0:
    #     Cbw=R_B*Cn_B/(Nbat*Q_lifetime*np.sqrt(ef_bat) );
    # else:
    #     Cbw=0;
    
    
    #  DG Fix cost
    cc_gen=b*Pn_DG*C_fuel+R_DG*Pn_DG/TL_DG+MO_DG;
    
    # from time import process_time
    # start = process_time()

    # reshape Cbuy
    global Cbuy
    Cbuy_reshaped = np.tile(Cbuy, (nVar, 1))

    (Eb, Pdg, Edump, Ens, Pch, Pdch, Pbuy, Psell, Pinv) =\
        energy_management(Ppv,Pwt,Eload_vector,Cn_B,Nbat,Pn_DG,(nVar, NT),
                          SOC_max,SOC_min,SOC_initial,
                          n_I,Grid,Cbuy_reshaped,a,Cn_I,LR_DG,C_fuel,Pbuy_max,Psell_max,cc_gen,Cbw,
                          self_discharge_rate,alfa_battery,c,k,Imax,Vnom,ef_bat)
    
    # print(process_time()-start)

    q=(a*Pdg+b*Pn_DG)*(Pdg>0);   # Fuel consumption of a diesel generator 

    #%% installation and operation cost

    #reset shapes
    Npv=np.around(X[:,0]);      # PV numbers
    Nwt=np.around(X[:,1]);      # WT numbers
    Nbat=np.around(X[:,2]);     # Battery pack numbers
    N_DG=np.around(X[:,3]);     # number of Diesel Generator
    Cn_I=X[:,4];                   # Inverter Capacity

    Pn_PV=Npv*Ppv_r;   # PV Total Capacity
    Pn_WT=Nwt*Pwt_r;   # WT Total Capacity
    Cn_B=Nbat*Cbt_r;   # Battery Total Capacity
    Pn_DG=N_DG*Cdg_r;  # Diesel Total Capacity
    
    # Total Investment cost ($)
    I_Cost=C_PV*Pn_PV + C_WT*Pn_WT+ C_DG*Pn_DG+C_B*Cn_B+C_I*Cn_I +C_CH;
    
    Top_DG=np.sum(Pdg>0)+1;
    L_DG=TL_DG/Top_DG;
    RT_DG=np.ceil(n/L_DG)-1; #Replecement time
    
    #Total Replacement cost ($)
    RC_PV= np.zeros(n);
    RC_WT= np.zeros(n);
    RC_DG= np.zeros(n);
    RC_B = np.zeros(n);
    RC_I = np.zeros(n);
    RC_CH = np.zeros(n);

    # we only want to do stuff if np.arange returns an actual range
    # then we need to resahpe Pn_PV[:,0] to tile it to the num of vals in the range
    # need to reshape the range to tile it by 50
    # then element wise pow

    # alt, store the arange in the correct range
    # then reshape and do pow

    RC_PV[np.arange(L_PV+1,n,L_PV)] = np.arange(1.001*L_PV,n,L_PV)
    RC_WT[np.arange(L_WT+1,n,L_WT)] = np.arange(1.001*L_WT,n,L_WT)
    RC_DG[np.arange(L_DG+1,n,L_DG).astype(np.int32)] = np.arange(1.001*L_DG,n,L_DG)
    RC_B[np.arange(L_B+1,n,L_B)] = np.arange(1.001*L_B,n,L_B)
    RC_I[np.arange(L_I+1,n,L_I)] = np.arange(1.001*L_I,n,L_I)
    RC_CH[np.arange(L_CH+1,n,L_CH)] = np.arange(1.001*L_CH,n,L_CH)
    
    RC_PV = np.swapaxes(np.tile(RC_PV, (nVar,1)), 0, 1)
    RC_WT = np.swapaxes(np.tile(RC_WT, (nVar,1)), 0, 1)
    RC_DG = np.swapaxes(np.tile(RC_DG, (nVar,1)), 0, 1)
    RC_B = np.swapaxes(np.tile(RC_B, (nVar,1)), 0, 1)
    RC_I = np.swapaxes(np.tile(RC_I, (nVar,1)), 0, 1)
    RC_CH = np.swapaxes(np.tile(RC_CH, (nVar,1)), 0, 1)

    RC_PV = R_PV*np.tile(Pn_PV, (n, 1))/(1+ir)**RC_PV
    RC_WT = R_WT*np.tile(Pn_WT, (n,1))/(1+ir)**RC_WT
    RC_DG = R_DG*np.tile(Pn_DG, (n,1))/(1+ir)**RC_DG
    RC_B = R_B*np.tile(Cn_B, (n,1))/(1+ir)**RC_B
    RC_I = R_I*np.tile(Cn_I, (n,1))/(1+ir)**RC_I
    RC_CH = np.tile(R_CH, nVar) /(1+ir)**RC_CH

    # RC_PV[np.arange(L_PV+1,n,L_PV)]= R_PV*Pn_PV[:,0]/(1+ir)**(np.arange(1.001*L_PV,n,L_PV));
    # RC_WT[np.arange(L_WT+1,n,L_WT)]= R_WT*Pn_WT[:,0]/(1+ir)** (np.arange(1.001*L_WT,n,L_WT)) ;
    # RC_DG[np.arange(L_DG+1,n,L_DG).astype(np.int32)]= R_DG*Pn_DG[:,0]/(1+ir)**(np.arange(1.001*L_DG,n,L_DG)) ;
    # RC_B[np.arange(L_B+1,n,L_B)] = R_B*Cn_B[:,0] /(1+ir)**(np.arange(1.001*L_B,n,L_B)) ;
    # RC_I[np.arange(L_I+1,n,L_I)] = R_I*Cn_I[:,0] /(1+ir)**(np.arange(1.001*L_I,n,L_I)) ;
    # RC_CH[np.arange(L_CH+1,n,L_CH)]  = np.tile(R_CH, nVar) /(1+ir)**(np.arange(1.001*L_CH,n,L_CH)) ;
    R_Cost=RC_PV+RC_WT+RC_DG+RC_B+RC_I+RC_CH;
    
    #Total
    #  M&O Cost ($/year)
    MO_Cost=( MO_PV*np.tile(Pn_PV, (n, 1))+ MO_WT*np.tile(Pn_WT, (n,1))+ MO_DG*np.sum(np.tile(Pn_DG, (n,1))>0)+ \
             MO_B*np.tile(Cn_B, (n,1))+ MO_I*np.tile(Cn_I, (n,1))+MO_CH)/(1+ir)**np.swapaxes(np.tile(np.array(range(n)), (nVar,1)),0,1)
    
    # DG fuel Cost
    C_Fu= np.tile(np.sum(C_fuel*q, axis=1)/(1+ir), (n,1))**np.swapaxes(np.tile(np.array(range(n)), (nVar, 1)), 0,1)


    # Salvage
    L_rem=(RT_PV+1)*L_PV-n; 
    S_PV=(R_PV*Pn_PV)*L_rem/L_PV * 1/(1+ir)**n # PV
    L_rem=(RT_WT+1)*L_WT-n;
    S_WT=(R_WT*Pn_WT)*L_rem/L_WT * 1/(1+ir)**n # WT
    L_rem=(RT_DG+1)*L_DG-n; 
    S_DG=(R_DG*Pn_DG)*L_rem/L_DG * 1/(1+ir)**n # DG
    L_rem=(RT_B +1)*L_B-n; 
    S_B =(R_B*Cn_B)*L_rem/L_B * 1/(1+ir)**n;
    L_rem=(RT_I +1)*L_I-n; 
    S_I =(R_I*Cn_I)*L_rem/L_I * 1/(1+ir)**n;
    L_rem=(RT_CH +1)*L_CH-n; 
    S_CH =(R_CH)*L_rem/L_CH * 1/(1+ir)**n;
    Salvage=S_PV+S_WT+S_DG+S_B+S_I+S_CH;


    #Emissions produced by Disesl generator (g)
    DG_Emissions=np.sum( q*(CO2 +NOx +SO2) )/1000;           # total emissions (kg/year)
    Grid_Emissions= np.sum( Pbuy*(E_CO2+E_SO2+E_NOx) )/1000; # total emissions (kg/year)
    
    Grid_Cost= (np.sum(Pbuy*Cbuy)-np.sum(Psell*Csell) )* 1/(1+ir)**np.array(range(1,n+1));
    
    #Capital recovery factor
    CRF=ir*(1+ir)**n/((1+ir)**n -1);
    
    # Totall Cost
    NPC=I_Cost+np.sum(R_Cost)+ np.sum(MO_Cost)+np.sum(C_Fu) -Salvage+np.sum(Grid_Cost);
    
    Operating_Cost=CRF*(np.sum(R_Cost)+ np.sum(MO_Cost)+np.sum(C_Fu) -Salvage+np.sum(Grid_Cost));


    LCOE = np.where(np.sum(Eload_vector-Ens, axis=1)>1, CRF*NPC/np.sum(Eload_vector-Ens+Psell, axis=1), 100)
    LEM = np.where(np.sum(Eload_vector-Ens, axis=1)>1, (DG_Emissions+Grid_Emissions)/np.sum(Eload_vector-Ens, axis=1), 100)
    # if np.sum(Eload_vector-Ens, axis=1)>1:
    #     LCOE=CRF*NPC/np.sum(Eload-Ens+Psell);                #Levelized Cost of Energy ($/kWh)
    #     LEM=(DG_Emissions+Grid_Emissions)/sum(Eload-Ens);    #Levelized Emissions(kg/kWh)
    # else:
    #     LCOE=100;
    #     LEM=100;
    
    LPSP=np.sum(Ens)/np.sum(Eload);   
    
    RE=1-np.sum(Pdg+Pbuy)/np.sum(Eload+Psell-Ens);
    if(np.isnan(RE)):
        RE=0;

    Z=LCOE+EM*LEM+10*(LPSP>LPSP_max)+10*(RE<RE_min)+100*(I_Cost>Budget)+\
        100*max(0, LPSP-LPSP_max)+100*max(0, RE_min-RE)+100*np.fmax(0, I_Cost-Budget);

    return Z
