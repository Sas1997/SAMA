
import numpy as np
import pandas as pd
from Input_Data import parameter
from EMS import energy_management
from Battery_Model import battery_model


def my_range(start,end,step):
    start=int(start)
    end=int(end)
    step=int(step)
    idx=[]
    for i in range(start,end,step):
        idx.append(i)
    return idx
    
    
    
# takes about 7 secs to run    
def fitness(X,Eload, G, T, Vw, inputs):
    NT = Eload.size # time step numbers

    Npv = np.round(X[0]) # PV number
    Nwt = np.round(X[1]) # WT number
    Nbat = np.round(X[2]) # Battery pack number
    N_DG = np.round(X[3]) # number of diesel generator
    Cn_I = X[4] # inverter capacity
    
    Pn_PV=Npv*inputs.Ppv_r     # PV Total Capacity
    Pn_WT=Nwt*inputs.Pwt_r     # WT Total Capacity
    Cn_B=Nbat*inputs.Cbt_r     # Battery Total Capacity
    Pn_DG=N_DG*inputs.Cdg_r    # Diesel Total Capacity
    
    # %% PV power calculation
    Tc = T+(((inputs.Tnoct-20)/800)*G) # Module Temprature
    Ppv = inputs.fpv*Pn_PV*(G/inputs.Gref)*(1+inputs.Tcof*(Tc-inputs.Tref)) # output power(kw)_hourly
    
    #%% Wind turbine Power Calculation
    v1=Vw # hourly wind speed
    v2 = ((inputs.h_hub / inputs.h0) ** inputs.alfa_wind_turbine) * v1 # v1 is the speed at a reference height;v2 is the speed at a hub height h2
    
    Pwt = np.zeros((8760,))
    for t in range(Pwt.size):
        if v2[t] < inputs.v_cut_in or v2[t] > inputs.v_cut_out:
            Pwt[t] = 0
        elif inputs.v_cut_in <= v2[t] and v2[t] < inputs.v_rated:
            Pwt[t] = v2[t]**3 * (inputs.Pwt_r / (inputs.v_rated**3 - inputs.v_cut_in)) - (inputs.v_cut_in**3 / (inputs.v_rated**3 - inputs.v_cut_in**3)) * inputs.Pwt_r
        elif inputs.v_rated <= v2[t] and v2[t] < inputs.v_cut_out:
            Pwt[t] = inputs.Pwt_r
        else:
            Pwt[t] = 0
        Pwt[t] = Pwt[t] * Nwt
    
    # %% Energy management
    # Battery wear cost
    if Cn_B > 0:
        Cbw = inputs.R_B * Cn_B / (Nbat * inputs.Q_lifetime * np.sqrt(inputs.ef_bat))
    else:
        Cbw = 0
    
    # DG fix cost
    cc_gen = inputs.b * Pn_DG * inputs.C_fuel + inputs.R_DG * Pn_DG / inputs.TL_DG + inputs.MO_DG
    
    (Eb, Pdg, Edump, Ens, Pch, Pdch, Pbuy, Psell, Pinv) = energy_management(Ppv, Pwt, Eload,
      Cn_B, Nbat, Pn_DG, NT, Cn_I, cc_gen, Cbw, inputs)
    
    
    
    Pdg = np.where(Pdg > 0, 1, 0)
    q = ((inputs.a * Pdg) + (inputs.b * Pn_DG)) * (Pdg) # fuel consumption of a diesel generator
    
    
    # installation and operation cost
    # total investment cost ($)
    I_Cost=inputs.C_PV*Pn_PV + inputs.C_WT*Pn_WT+ inputs.C_DG*Pn_DG+inputs.C_B*Cn_B+inputs.C_I*Cn_I +inputs.C_CH
    
    Top_DG = np.count_nonzero(Pdg) + 1
    L_DG = inputs.TL_DG / Top_DG
    RT_DG = np.ceil(inputs.n / L_DG) - 1 
    
    # total replacement cost ($)
    
    RC_PV= np.zeros((inputs.n+1))
    RC_WT= np.zeros((inputs.n+1))
    RC_DG= np.zeros((inputs.n+1))
    RC_B = np.zeros((inputs.n+1))
    RC_I = np.zeros((inputs.n+1))
    RC_CH = np.zeros((inputs.n+1))
    
    
    res_idx_L_PV=my_range(inputs.L_PV,inputs.n,inputs.L_PV)
    res_idx_L_WT=my_range(inputs.L_WT,inputs.n,inputs.L_WT)
    res_idx_L_DG=my_range(int(L_DG),inputs.n,int(L_DG))
    res_idx_L_B=my_range(inputs.L_B,inputs.n,inputs.L_B)
    res_idx_L_CH=my_range(inputs.L_CH,inputs.n,inputs.L_CH)
    res_idx_L_I=my_range(inputs.L_I,inputs.n,inputs.L_I)
    
    # TODO:replacement cost
    if(len(res_idx_L_PV)>0):
        RC_PV[res_idx_L_PV]= inputs.R_PV*Pn_PV / (1+inputs.ir) ** (((1.001*inputs.L_PV)-inputs.L_PV)+ np.array(my_range(inputs.L_PV,inputs.n,inputs.L_PV)))
    if(len(res_idx_L_WT)>0):
        RC_WT[res_idx_L_WT]= inputs.R_WT*Pn_WT / (1+inputs.ir) **  (((1.001*inputs.L_WT)-inputs.L_WT)+ np.array(my_range(inputs.L_WT,inputs.n,inputs.L_WT)))
    if(len(res_idx_L_DG)>0):
        RC_DG[res_idx_L_DG]= inputs.R_DG*Pn_DG / (1+inputs.ir) **  (((1.001*L_DG)-L_DG)+ np.array(my_range(L_DG,inputs.n,L_DG)))
    if(len(res_idx_L_B)>0):
        RC_B[res_idx_L_B] = inputs.R_B*Cn_B / (1+inputs.ir) **     (((1.001*inputs.L_B)-inputs.L_B)+ np.array(my_range(inputs.L_B,inputs.n,inputs.L_B)))
    if(len(res_idx_L_I)>0):
        RC_I[res_idx_L_I] = inputs.R_I*Cn_I / (1+inputs.ir) **     (((1.001*inputs.L_I)-inputs.L_I)+ np.array(my_range(inputs.L_I,inputs.n,inputs.L_I)))
    if(len(res_idx_L_CH)>0):    
        RC_CH[res_idx_L_CH] = inputs.R_CH / (1+inputs.ir) **       (((1.001*inputs.L_CH)-inputs.L_CH)+ np.array(my_range(inputs.L_CH,inputs.n,inputs.L_CH)))
    
    R_Cost=RC_PV+RC_WT+RC_DG+RC_B+RC_I+RC_CH
    
    # Total M&O Cost ($/year)
    MO_Cost=(inputs.MO_PV*Pn_PV + inputs.MO_WT*Pn_WT + \
             inputs.MO_DG*np.count_nonzero(Pn_DG)+ \
                 inputs.MO_B*Cn_B+ inputs.MO_I*Cn_I +inputs.MO_CH) / (1+inputs.ir)\
                 ** np.array(range(1,inputs.n+1))
    
    # DG fuel Cost
    C_Fu= sum(inputs.C_fuel*q)/(1+inputs.ir) ** np.array(range(1,inputs.n+1))
    
    # Salvage
    L_rem=(inputs.RT_PV+1)*inputs.L_PV-inputs.n
    
    S_PV=(inputs.R_PV*Pn_PV)*L_rem/inputs.L_PV * 1/(1+inputs.ir) ** inputs.n # PV
    L_rem=(inputs.RT_WT+1)*inputs.L_WT-inputs.n
    S_WT=(inputs.R_WT*Pn_WT)*L_rem/inputs.L_WT * 1/(1+inputs.ir) ** inputs.n # WT
    L_rem=(RT_DG+1)*L_DG-inputs.n
    S_DG=(inputs.R_DG*Pn_DG)*L_rem/L_DG * 1/(1+inputs.ir) ** inputs.n # DG
    L_rem=(inputs.RT_B +1)*inputs.L_B-inputs.n
    S_B =(inputs.R_B*Cn_B)*L_rem/inputs.L_B * 1/(1+inputs.ir) ** inputs.n
    L_rem=(inputs.RT_I +1)*inputs.L_I-inputs.n
    S_I =(inputs.R_I*Cn_I)*L_rem/inputs.L_I * 1/(1+inputs.ir) ** inputs.n
    L_rem=(inputs.RT_CH +1)*inputs.L_CH-inputs.n
    S_CH =(inputs.R_CH)*L_rem/inputs.L_CH * 1/(1+inputs.ir) ** inputs.n
    Salvage=S_PV+S_WT+S_DG+S_B+S_I+S_CH
    
    
    # Emissions produced by Disesl generator (g)
    DG_Emissions=sum(q*(inputs.CO2 + inputs.NOx + inputs.SO2))/1000 # total emissions (kg/year)
    Grid_Emissions= sum(Pbuy*(inputs.E_CO2+inputs.E_SO2+inputs.E_NOx))/1000 # total emissions (kg/year)
    
    Grid_Cost= (sum(Pbuy*inputs.Cbuy)-sum(Psell*inputs.Csell))* 1/(1+inputs.ir)** np.array(range(1,inputs.n+1))
    
    
    # Capital recovery factor
    CRF=inputs.ir*(1+inputs.ir)**inputs.n/((1+inputs.ir)**inputs.n -1)
    
    # Totall Cost
    NPC=I_Cost+sum(R_Cost)+sum(MO_Cost)+sum(C_Fu)-Salvage+sum(Grid_Cost)
    Operating_Cost=CRF*(sum(R_Cost)+ sum(MO_Cost)+sum(C_Fu)-Salvage+sum(Grid_Cost))
    
    if sum(Eload-Ens) > 1:
        LCOE=CRF*NPC/sum(Eload-Ens+Psell)                # Levelized Cost of Energy ($/kWh)
        LEM=(DG_Emissions+Grid_Emissions)/sum(Eload-Ens) # Levelized Emissions(kg/kWh)
    else:
        LCOE = 100
        LEM = 100
    
    LPSP = sum(Ens) / sum(Eload)
    
    RE=1-sum(Pdg+Pbuy)/sum(Eload+Psell-Ens)
    RE=np.nan_to_num(RE)
    
    Z=LCOE+inputs.EM*LEM+10*(LPSP>inputs.LPSP_max)+10*(RE<inputs.RE_min)+100*(I_Cost>inputs.Budget)+\
        100*max(0, LPSP-inputs.LPSP_max)+100*max(0, inputs.RE_min-RE)+100*max(0, I_Cost-inputs.Budget);
    return Z
