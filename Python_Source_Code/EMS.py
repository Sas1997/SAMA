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
    # Eb=np.zeros(NT+1);
    # Pch=np.zeros(NT);
    # Pdch=np.zeros(NT);
    # Ech=np.zeros(NT);
    # Edch=np.zeros(NT);
    # Pdg=np.zeros(NT);
    # Edump=np.zeros(NT);
    # Ens=np.zeros(NT);
    # Psell=np.zeros(NT);
    # Pbuy=np.zeros(NT);
    # Pinv=np.zeros(NT);
    # Ebmax=SOC_max*Cn_B;
    # Ebmin=SOC_min*Cn_B;
    # Eb[0]=SOC_initial*Cn_B;
    # dt=1;
    
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

    Eb_e = np.where(load_greater, (Ebmax-Eb)/ef_bat,    #Battery charge power calculated based on surEloadus energy and battery empty  capacity
        np.where(case1, (Eb-Ebmin)*ef_bat,
            np.where(case2, (Eb-Ebmin)*ef_bat,
                np.where(case3, (Eb-Ebmin)*ef_bat,
                    np.where(case4, (Eb-Ebmin)*ef_bat,
                        np.where(case5, (Eb[t]-Ebmin)*ef_bat,
                            np.where(case6, (Eb[t]-Ebmin)*ef_bat)))))))
    
    Pch = np.where(load_greater, np.amin(np.amin(Eb_e, P_RE-Eload/n_i, axis=0), Pch_max, axis=0), Pch) # Battery maximum charge power limit

    Edef_AC = np.where(np.logical_not(load_greater), Eload-np.amin(Pinv_max, n_I*P_RE, axis=0), 0) 
    
    # need to do pbuy before this case2
    Edef_AC = np.where(case2, Eload-Pbuy-np.amin(Pinv_max, n_I*(P_RE+Pdch), axis=0), 
        np.where(case5, Eload-np.amin(Pinv_max, n_I*(P_RE+Pdch), axis=0), 
            np.where(case6, Eload-np.amin(Pinv_max, n_I*(P_RE+Pdch), axis=0), Edef_AC)))

    Pdg_intermediate = np.where(case1, np.amin(Edef_AC-np.amin(Edef_AC, Pbuy_max, axis=0), Pn_DG, axis=0),
        np.where(case2, np.amin(Edef_AC, Pbuy_max, axis=0),
            np.where(np.logical_or(np.logical_or(case3, case4), case5), np.amin(Edef_AC,Pn_DG, axis=0),
                np.where(case6, np.amin(Edef_AC-Pbuy,Pn_DG, axis=0), Pdg))))
    Pdg = np.where(case1, Pdg_intermediate*(Pdg_intermediate>=LR_DG*Pn_DG)+LR_DG*Pn_DG*(Pdg_intermediate<LR_DG*Pn_DG)*(Pdg_intermediate>Pdg_min),
        np.where(case2, Pdg_intermediate*(Pdg_intermediate>=LR_DG*Pn_DG)+LR_DG*Pn_DG*(Pdg_intermediate<LR_DG*Pn_DG)*(Pdg_intermediate>Pdg_min),
            np.where(case3, Pdg_intermediate*(Pdg_intermediate>=LR_DG*Pn_DG)+LR_DG*Pn_DG*(Pdg_intermediate<LR_DG*Pn_DG)*(Pdg_intermediate>Pdg_min)),
                np.where(case4, Pdg_intermediate*(Pdg_intermediate>=LR_DG*Pn_DG)+LR_DG*Pn_DG*(Pdg_intermediate<LR_DG*Pn_DG)*(Pdg_intermediate>Pdg_min),
                    np.where(case5, Pdg_intermediate*(Pdg_intermediate>=LR_DG*Pn_DG)+LR_DG*Pn_DG*(Pdg_intermediate<LR_DG*Pn_DG)*(Pdg_intermediate>Pdg_min),
                        np.where(case6, Pdg_intermediate*(Pdg_intermediate>=LR_DG*Pn_DG)+LR_DG*Pn_DG*(Pdg_intermediate<LR_DG*Pn_DG)*(Pdg_intermediate>Pdg_min), Pdg)))))

    Edef_AC = np.where(case1, Eload-Pdg-np.amin(Edef_AC,Pbuy_max, axis=0)-np.amin(Pinv_max, n_I*P_RE, axis=0),
        np.where(case3, Eload-Pdg-np.amin(Pinv_max, n_I*(P_RE+Pdch),  axis=0), 
            np.where(case4, Eload-Pdg-np.amin(Pinv_max, n_I*(P_RE+Pdch), axis=0), Edef_AC)))

    # do edef_dc here
    Edef_DC = np.where(case1, Edef_AC/n_I*(Edef_AC>0),
        np.where(case2, (Eload[t]-Pbuy[t])/n_I-P_RE(t),
            np.where(case3, np.where(Eload[t]-Pbuy(t)-Pdg[t])/n_I-P_RE[t]) > 0,  )))

    Psur_AC = np.where(load_greater, n_I*(P_RE-Pch-Eload), inf) #surplus Energy
    Psell = np.where(load_greater, np.amin(np.amin(Psur_AC, Psell_max, axis=0), np.amax(0, Pinv_max-Eload, axis=0), axis=0),
        np.where(case3, np.amax(0, np.amin(Pdg-Edef_AC,Psell_max, axis=0), axis=0),
            np.where(case4, np.amax(0, np.amin(-Edef_AC,Psell_max, axis=0), axis=0), 
                np.where(case5, np.amax(0, np.amin(Pd-Edef_AC,Psell_max, axis=0), axis=0), Psell))))

    Edump = np.where(load_greater, P_RE-Pch-(Eload+Psell)/n_I, Edump)

    # return Eb, Pdg, Edump, Ens, Pch, Pdch, Pbuy, Psell, Pinv

    else:

        if (Cbuy[t]<= price_dg) and (price_dg<=Cbw): # Grid, DG , Bat : 1
            Pdg[t]=min(Edef_AC-min(Edef_AC,Pbuy_max),Pn_DG);
            Pdg[t]=Pdg[t]*(Pdg[t]>=LR_DG*Pn_DG)+LR_DG*Pn_DG*(Pdg[t]<LR_DG*Pn_DG)*(Pdg[t]>Pdg_min);
            
            Edef_AC=Eload[t]-Pdg(t)-min(Edef_AC,Pbuy_max)-min(Pinv_max, n_I*P_RE[t]);
            Edef_DC=Edef_AC/n_I*(Edef_AC>0);
            Eb_e=(Eb[t]-Ebmin)*ef_bat;
            Pdch[t]= min( Eb_e,Edef_DC);
            Pdch[t]= min(Pdch[t],Pdch_max);
            
            Esur_AC=-Edef_AC*(Edef_AC<0);
            Pbuy[t]=Pbuy[t]-Esur_AC*(Grid==1);


        elif (Cbuy[t]<= Cbw) and (Cbw<price_dg):  #Grid, Bat , DG : 2
            print(22)
            Pbuy[t]=min(Edef_AC,Pbuy_max);
            
            Edef_DC=(Eload[t]-Pbuy[t])/n_I-P_RE(t);
            Eb_e=(Eb[t]-Ebmin)*ef_bat;
            Pdch[t]= min( Eb_e,Edef_DC);
            Pdch[t]=min(Pdch[t],Pdch_max);
            
            Edef_AC=Eload[t]-Pbuy[t]-min(Pinv_max, n_I*(P_RE[t]+Pdch[t]));
            Pdg[t]=min(Edef_AC,Pn_DG);
            Pdg[t]=Pdg[t]*(Pdg[t]>=LR_DG*Pn_DG)+LR_DG*Pn_DG*(Pdg[t]<LR_DG*Pn_DG)*(Pdg[t]>Pdg_min);


        elif (price_dg<Cbuy[t]) and (Cbuy[t]<=Cbw): #DG, Grid , Bat :3
            print(23)
            Pdg[t]=min(Edef_AC,Pn_DG);
            Pdg[t]=Pdg(t)*(Pdg[t]>=LR_DG*Pn_DG)+LR_DG*Pn_DG*(Pdg[t]<LR_DG*Pn_DG)*(Pdg[t]>Pdg_min);
            
            Pbuy[t]=max(0,  min(Edef_AC-Pdg[t],Pbuy_max) );
            Psell[t]=max(0, min(Pdg[t]-Edef_AC,Psell_max) );
            
            Edef_DC=(Eload[t]-Pbuy(t)-Pdg[t])/n_I-P_RE[t];
            Edef_DC=Edef_DC*(Edef_DC>0);
            Eb_e=(Eb[t]-Ebmin)*ef_bat;
            Pdch[t]= min( Eb_e,Edef_DC);
            Pdch[t]=min(Pdch[t],Pdch_max);
        elif (price_dg<Cbw) and (Cbw<Cbuy[t]):  #DG, Bat , Grid :4
            print(24)  
            Pdg[t]=min(Edef_AC,Pn_DG);
            Pdg[t]=Pdg[t]*(Pdg[t]>=LR_DG*Pn_DG)+LR_DG*Pn_DG*(Pdg[t]<LR_DG*Pn_DG)*(Pdg[t]>Pdg_min);    
            
            Edef_DC=(Eload[t]-Pdg[t])/n_I-P_RE[t];
            Edef_DC=Edef_DC*(Edef_DC>0);
            Eb_e=(Eb[t]-Ebmin)*ef_bat;
            Pdch[t]= min( Eb_e,Edef_DC);  
            Pdch[t]= min(Pdch[t],Pdch_max);
            
            Edef_AC=Eload[t]-Pdg[t]-min(Pinv_max, n_I*(P_RE[t]+Pdch[t]) );
            Pbuy[t]=max(0,  min(Edef_AC,Pbuy_max) );
            Psell[t]=max(0, min(-Edef_AC,Psell_max) );
            
        elif (Cbw<price_dg) and (price_dg<Cbuy[t]):  #Bat ,DG, Grid :5
            print(25)
            Edef_DC=Eload[t]/n_I-P_RE[t];
            Eb_e=(Eb[t]-Ebmin)*ef_bat;
            Pdch[t]=min( Eb_e,Edef_DC);  
            Pdch[t]=min(Pdch[t],Pdch_max);  
            
            Edef_AC=Eload[t]-min(Pinv_max, n_I*(P_RE[t]+Pdch[t]) );
            Pdg[t]=min(Edef_AC,Pn_DG);
            Pdg[t]=Pdg[t]*(Pdg[t]>=LR_DG*Pn_DG)+LR_DG*Pn_DG*(Pdg[t]<LR_DG*Pn_DG)*(Pdg[t]>Pdg_min); 
            
            Pbuy[t]=max(0,  min(Edef_AC-Pdg[t],Pbuy_max) );
            Psell[t]=max(0, min(Pdg[t]-Edef_AC,Psell_max) );
        else: #Bat , Grid , DG: 6
            Edef_DC=min(Pinv_max, Eload[t]/n_I)-P_RE[t];
            Eb_e=(Eb[t]-Ebmin)*ef_bat;
            Pdch[t]=min( Eb_e,Edef_DC)*(Edef_DC>0);  
            Pdch[t]=min( Pdch[t],Pdch_max ); 
            
            Edef_AC=Eload[t]-min(Pinv_max, n_I*(P_RE[t]+Pdch[t]) );
            Pbuy[t]= min( Edef_AC, Pbuy_max);
            
            Pdg[t]=min(Edef_AC-Pbuy[t],Pn_DG);
            Pdg[t]=Pdg[t]*(Pdg[t]>=LR_DG*Pn_DG)+LR_DG*Pn_DG*(Pdg[t]<LR_DG*Pn_DG)*(Pdg[t]>Pdg_min); 
            
    
        
        Edef_DC=(Eload[t]+Psell[t]-Pdg[t]-Pbuy[t])/n_I-(P_RE[t]+Pdch[t]-Pch[t]);
        if Edef_DC<0:
            Eb_e=(Ebmax-Eb[t])/ef_bat;
            Pch[t]=min(Eb_e, Pch[t]-Edef_DC);
            Pch[t]=min(Pch[t],Pch_max);

        Esur=Eload[t]+Psell[t]-Pbuy[t]-Pdg[t]-min(Pinv_max, (P_RE[t]+Pdch[t]-Pch[t])*n_I);  
        Ens[t]=Esur*(Esur>0);
        Edump[t]=-Esur*(Esur<0);

    
    #%% Battery charging and discharging energy is determined based on charging and discharging power and the battery charge level is updated.
    Ech[t]=Pch[t]*dt;
    Edch[t]=Pdch[t]*dt;
    # index out of bounds error check
    Eb[t+1]=(1-self_discharge_rate)*Eb[t]+ef_bat*Ech[t]-Edch[t]/ef_bat;
    
    # print(process_time()-start)
    return np.array(Eb), np.array(Pdg), np.array(Edump), np.array(Ens), np.array(Pch), np.array(Pdch), np.array(Pbuy), np.array(Psell), np.array(Pinv)







 