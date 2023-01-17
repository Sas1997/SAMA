import numpy as np
from Battery_Model import battery_model

"""
Energy management 
"""
def energy_management(Ppv,Pwt,Eload,Cn_B,Nbat,
                      Pn_DG,NT,SOC_max,SOC_min,
                      SOC_initial,n_I,Grid,Cbuy,
                      a,Pinv_max,LR_DG,C_fuel,Pbuy_max,
                      Psell_max,cc_gen,Cbw,self_discharge_rate,
                      alfa_battery,c,k,Imax,Vnom,ef_bat):
    #^^^^^^^^^^^^^^READ INPUTS^^^^^^^^^^^^^^^^^^
    Eb=np.zeros(NT);
    Pch=np.zeros(NT);
    Pdch=np.zeros(NT);
    Ech=np.zeros(NT);
    Edch=np.zeros(NT);
    Pdg=np.zeros(NT);
    Edump=np.zeros(NT);
    Ens=np.zeros(NT);
    Psell=np.zeros(NT);
    Pbuy=np.zeros(NT);
    Pinv=np.zeros(NT);
    Ebmax=SOC_max*Cn_B;
    Ebmin=SOC_min*Cn_B;
    Eb[0]=SOC_initial*Cn_B;
    dt=1;


    
    if Grid==0:
        Pbuy_max=0
        Psell_max=0
    
    # %%
    
    P_RE=Ppv+Pwt;
    Pdg_min=0.05*Pn_DG; # LR_DG

    Pdch_max, Pch_max = battery_model(Nbat, Eb, alfa_battery, c, k, Imax, Vnom, ef_bat) # kW    

    # numpy conditionals
    load_greater = np.logical_and(P_RE>=(Eload/n_I),(Eload<=Pinv_max))                                          #if PV+Pwt greater than load  (battery should charge)

    price_dg = cc_gen+a*C_fuel # DG cost ($/kWh)
    case1 = np.logical_and(np.logical_not(load_greater), np.logical_and(Cbuy <= price_dg, price_dg <= Cbw))     # Grid, DG , Bat : 1
    case2 = np.logical_and(np.logical_not(load_greater), np.logical_and(Cbuy <= Cbw, Cbw<price_dg))             #Grid, Bat , DG : 2
    case3 = np.logical_and(np.logical_not(load_greater), np.logical_and(price_dg<Cbuy, Cbuy<=Cbw))              #DG, Grid , Bat :3
    case4 = np.logical_and(np.logical_not(load_greater), np.logical_and(price_dg<Cbw, Cbw<Cbuy))                #DG, Bat , Grid :4
    case5 = np.logical_and(np.logical_not(load_greater), np.logical_and(Cbw<price_dg, price_dg<Cbuy))           #Bat ,DG, Grid :5
    case6 = np.logical_and(np.logical_not(load_greater),                                                        #Bat , Grid , DG: 6
        np.logical_and(np.logical_not(case1), 
            np.logical_and(np.logical_not(case2), 
                np.logical_and(np.logical_not(case3),
                    np.logical_and(np.logical_not(case4), np.logical_not(case5))))))

    #Battery charge power calculated based on surEloadus energy and battery empty  capacity
    Eb_e = np.where(load_greater, (Ebmax-Eb)/ef_bat, (Eb-Ebmin)*ef_bat)
    
    #if PV+Pwt greater than load  (battery should charge)
    #Battery charge power calculated based on surEloadus energy and battery empty  capacity 
    Pch = np.where(load_greater, np.fmin(np.fmin(Eb_e, P_RE-Eload/n_I), Pch_max), Pch) # Battery maximum charge power limit
    Psell = np.where(load_greater, np.fmin(np.fmin(n_I*(P_RE-Pch-Eload), Psell_max), np.fmax(0, Pinv_max-Eload)), Psell)
    Edump = np.where(load_greater, P_RE-Pch-(Eload+Psell)/n_I, Edump)

    #%% if load greater than PV+Pwt 
    Edef_AC = np.where(np.logical_not(load_greater), Eload-np.fmin(Pinv_max, n_I*P_RE), 0) 
    
    # Grid, DG , Bat : 1
    Pbuy = np.where(case1, np.fmin(Edef_AC, Pbuy_max), Pbuy)
    Pdg = np.where(case1, np.fmin(Edef_AC-Pbuy, Pn_DG), Pdg)
    Pdg = np.where(case1, Pdg*(Pdg>=LR_DG*Pn_DG)+LR_DG*Pn_DG*(Pdg<LR_DG*Pn_DG)*(Pdg>Pdg_min), Pdg)
    Edef_AC = np.where(case1, Eload-Pdg-Pbuy-np.fmin(Pinv_max, n_I*P_RE), Edef_AC)
    Edef_DC = np.where(case1, Edef_AC/n_I*(Edef_AC>0), np.fmin(Pinv_max, Eload/n_I)-P_RE)
    Pdch = np.where(case1, np.amin((np.amin((Eb_e, Edef_DC), axis=0), Pdch_max), axis=0), Pdch)
    Esur_AC = np.where(case1, -Edef_AC*(Edef_AC<0), Edef_AC)
    Pbuy = np.where(case1, Pbuy-Esur_AC*(Grid==1), Pbuy)

    #Grid, Bat , DG : 2
    Pbuy = np.where(case2, np.fmin(Edef_AC,Pbuy_max), Pbuy)
    Edef_DC = np.where(case2, (Eload-Pbuy)/n_I-P_RE, Edef_DC)
    Pdch = np.where(case2, np.amin((np.amin((Eb_e, Edef_DC), axis=0), Pdch_max), axis=0), Pdch)
    Edef_AC = np.where(case2, Eload-Pbuy-np.fmin(Pinv_max, n_I*(P_RE+Pdch)), Edef_AC)
    Pdg = np.where(case2, np.fmin(Edef_AC, Pn_DG), Pdg)
    Pdg = np.where(case2, Pdg*(Pdg>=LR_DG*Pn_DG)+LR_DG*Pn_DG*(Pdg<LR_DG*Pn_DG)*(Pdg>Pdg_min), Pdg)

    #DG, Grid , Bat :3
    Pdg = np.where(case3, np.fmin(Edef_AC, Pn_DG), Pdg)
    Pdg = np.where(case3, Pdg*(Pdg>=LR_DG*Pn_DG)+LR_DG*Pn_DG*(Pdg<LR_DG*Pn_DG)*(Pdg>Pdg_min), Pdg)
    Pbuy = np.where(case3, np.fmax(0, np.fmin(Edef_AC-Pdg, Pbuy_max)), Pbuy)
    Psell = np.where(case3, np.fmax(0, np.fmin(Pdg-Edef_AC, Psell_max)), Psell)
    Edef_DC = np.where(case3, (Eload-Pbuy-Pdg)/n_I-P_RE, Edef_DC)
    Edef_DC = np.where(case3, Edef_DC*(Edef_DC>0), Edef_DC)
    Pdch = np.where(case3, np.amin((np.amin((Eb_e, Edef_AC), axis=0), Pdch_max), axis=0), Pdch)

    #DG, Bat , Grid :4
    Pdg = np.where(case4, np.fmin(Edef_AC, Pn_DG), Pdg)
    Pdg = np.where(case4, Pdg*(Pdg>=LR_DG*Pn_DG)+LR_DG*Pn_DG*(Pdg<LR_DG*Pn_DG)*(Pdg>Pdg_min), Pdg)
    Edef_DC = np.where(case4, (Eload-Pdg)/n_I-P_RE, Edef_DC)
    Edef_DC = np.where(case4, Edef_DC*(Edef_DC>0), Edef_DC)
    Pdch = np.where(case4, np.amin((np.amin((Eb_e, Edef_DC), axis=0), Pdch_max), axis=0), Pdch)
    Edef_AC = np.where(case4, Eload-Pdg-np.fmin(Pinv_max, n_I*(P_RE+Pdch)), Edef_AC)
    Pbuy = np.where(case4, np.fmax(0, np.fmin(Edef_AC, Pbuy_max)), Pbuy)
    Psell = np.where(case5, np.fmax(0, np.fmin(-Edef_AC, Psell_max)), Psell)

    #Bat ,DG, Grid :5
    Edef_DC = np.where(case5, Eload/n_I-P_RE, Edef_DC)
    Pdch = np.where(case5, np.amin((Eb_e,Edef_DC), axis=0), Pdch)  
    Pdch = np.where(case5, np.amin((Pdch,Pdch_max), axis=0), Pdch)  
    Edef_AC = np.where(case5, Eload-np.fmin(Pinv_max, n_I*(P_RE+Pdch)), Edef_AC)
    Pdg = np.where(case5, np.fmin(Edef_AC,Pn_DG), Pdg)
    Pdg = np.where(case5, Pdg*(Pdg>=LR_DG*Pn_DG)+LR_DG*Pn_DG*(Pdg<LR_DG*Pn_DG)*(Pdg>Pdg_min), Pdg) 
    Pbuy = np.where(case5, np.fmax(0, np.fmin(Edef_AC-Pdg,Pbuy_max)), Pbuy)
    Psell = np.where(case5, np.fmax(0, np.fmin(Pdg-Edef_AC,Psell_max)), Psell)

    #Bat , Grid , DG: 6
    Edef_DC = np.where(case6, np.fmin(Pinv_max, Eload/n_I)-P_RE, Edef_DC)
    Pdch = np.where(case6, np.amin((Eb_e,Edef_DC), axis=0)*(Edef_DC>0), Pdch)  
    Pdch = np.where(case6, np.amin((Pdch,Pdch_max),axis=0), Pdch) 
    Edef_AC = np.where(case6, Eload-np.fmin(Pinv_max, n_I*(P_RE+Pdch)), Edef_AC)
    Pbuy = np.where(case6, np.fmin(Edef_AC, Pbuy_max), Pbuy)
    Pdg = np.where(case6, np.fmin(Edef_AC-Pbuy,Pn_DG), Pdg)
    Pdg = np.where(case6, Pdg*(Pdg>=LR_DG*Pn_DG)+LR_DG*Pn_DG*(Pdg<LR_DG*Pn_DG)*(Pdg>Pdg_min), Pdg) 

    Edef_DC = np.where(np.logical_not(load_greater), (Eload+Psell-Pdg-Pbuy)/n_I-(P_RE+Pdch-Pch), Edef_DC)
    Eb_e = np.where(np.logical_and(np.logical_not(load_greater), Edef_DC < 0), (Ebmax-Eb)/ef_bat, Eb_e)
    Pch = np.where(np.logical_and(np.logical_not(load_greater), Edef_DC < 0), np.amin((np.amin((Eb_e, Pch-Edef_DC), axis=0), Pch_max), axis=0), Pch)
    Esur = np.where(np.logical_not(load_greater), Eload+Psell-Pbuy-Pdg-np.fmin(Pinv_max, (P_RE+Pdch-Pch)*n_I), 0)
    Ens = np.where(np.logical_not(load_greater), Esur*(Esur>0), Ens)
    Edump = np.where(np.logical_not(load_greater), -Esur*(Esur<0), Edump)

    Ech = Pch * dt
    Edch = Pdch * dt
    
    return Eb, Pdg, Edump, Ens, Pch, Pdch, Pbuy, Psell, Pinv