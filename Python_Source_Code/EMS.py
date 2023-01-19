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
    
    P_RE=Ppv+Pwt    # (8760,)
    Pdg_min=0.05*Pn_DG; # LR_DG # float

    Eb=[SOC_initial*Cn_B]
    Pch=[]
    Pdch=[]
    Ech=[]
    Edch=[]
    Pdg=[]
    Edump=[]
    Ens=[]
    Psell=[]
    Pbuy=[]
    Pinv=[]
    Ebmax=SOC_max*Cn_B;
    Ebmin=SOC_min*Cn_B;
    dt=1;

    # define cases
    load_greater = np.logical_and(P_RE>=(Eload/n_I),(Eload<=Pinv_max))
    price_dg = cc_gen+a*C_fuel # DG cost ($/kWh)  
    case1 = np.logical_and(np.logical_not(load_greater), np.logical_and(Cbuy <= price_dg, price_dg <= Cbw))     # Grid, DG , Bat : 1
    case2 = np.logical_and(np.logical_not(load_greater), np.logical_and(Cbuy <= Cbw, Cbw<price_dg))             #Grid, Bat , DG : 2
    case3 = np.logical_and(np.logical_not(load_greater), np.logical_and(price_dg<Cbuy, Cbuy<=Cbw))              #DG, Grid , Bat :3
    case4 = np.logical_and(np.logical_not(load_greater), np.logical_and(price_dg<Cbw, Cbw<Cbuy))                #DG, Bat , Grid :4
    case5 = np.logical_and(np.logical_not(load_greater), np.logical_and(Cbw<price_dg, price_dg<Cbuy))
    t = 0
    while t < NT:
        if t == 0:
            Eb_curr = SOC_initial*Cn_B
        else:
            Eb_curr = (1-self_discharge_rate)*Eb_curr+ef_bat*Ech[-1]-Edch[-1]/ef_bat

        Pdch_max, Pch_max = battery_model(Nbat, Eb_curr, alfa_battery, c, k, Imax, Vnom, ef_bat)
        Pch_curr, Pdch_curr, Ech_curr, Edch_curr, Pdg_curr, Edump_curr, Ens_curr, Psell_curr, Pbuy_curr, Pinv_curr = 0, 0, 0, 0, 0, 0, 0, 0, 0, 0

        if load_greater[t]:
            Eb_e = (Ebmax - Eb_curr) / ef_bat
            Pch_curr = min(min(Eb_e, P_RE[t] - Eload[t]/n_I), Pch_max)
            Psur_AC = n_I * (P_RE[t]-Pch_curr-Eload[t])
            Psell_curr = min(min(Psur_AC, Psell_max), max(0, Pinv_max-Eload[t]))
            Edump_curr = P_RE[t]-Pch_curr - (Eload[t] + Psell_curr) / n_I
        
        else:
            Edef_AC = Eload[t] - min(Pinv_max, n_I*P_RE[t])
            price_dg = cc_gen + a * C_fuel

            if case1[t]:
                Pbuy_curr = min(Edef_AC, Pbuy_max)
                Pdg_curr = min(Edef_AC-Pbuy_curr, Pn_DG)
                Pdg_curr = Pdg_curr * (Pdg_curr >= LR_DG * Pn_DG) + LR_DG * Pn_DG * (Pdg_curr < LR_DG*Pn_DG) * (Pdg_curr > Pdg_min)
                Edef_AC = Edef_AC - Pdg_curr - Pbuy_curr
                Edef_DC = Edef_AC/n_I * (Edef_AC > 0)
                Eb_e = (Eb_curr - Ebmin) * ef_bat
                Pdch_curr = min(min(Eb_e, Edef_DC), Pdch_max)
                Esur_AC = -Edef_AC * (Edef_AC < 0)
                Pbuy_curr = Pbuy_curr - Esur_AC * (Grid)
            
            elif case2[t]:
                Pbuy_curr = min(Edef_AC, Pbuy_max)
                Edef_DC = (Eload[t] - Pbuy_curr)/n_I - P_RE[t]
                Eb_e = (Eb_curr - Ebmin) * ef_bat
                Pdch_curr = min(min(Eb_e, Edef_DC), Pdch_max)
                Edef_AC = Eload[t]-Pbuy_curr-min(Pinv_max, n_I*(P_RE[t]+Pdch_curr))
                Pdg_curr = min(Edef_AC, Pn_DG)
                Pdg_curr = Pdg_curr * (Pdg_curr>=LR_DG*Pn_DG)+ LR_DG * Pn_DG * (Pdg_curr<LR_DG*Pn_DG) * (Pdg_curr>Pdg_min)

            elif case3[t]:
                Pdg_curr = min(Edef_AC, Pn_DG)
                Pdg_curr = Pdg_curr*(Pdg_curr>=LR_DG*Pn_DG)+LR_DG*Pn_DG*(Pdg_curr<LR_DG*Pn_DG)*(Pdg_curr>Pdg_min)
                Pbuy_curr = max(0, min(Edef_AC-Pdf_curr, Pbuy_max))
                Psell_curr = max(0, min(Pdg_curr - Edef_AC, Psell_max))
                Edef_DC = (Eload[t]-Pbuy_curr-Pdg_curr)/n_I-P_RE[t]
                Edef_DC = Edef_DC * (Edef_DC > 0)
                Eb_e = (Eb_curr - Ebmin) * ef_bat
                Pdch_curr = min(min(Eb_e, Edef_DC), Pdch_max)

            elif case4[t]:
                Pdg_curr=min(Edef_AC,Pn_DG);
                Pdg_curr=Pdg_curr*(Pdg_curr>=LR_DG*Pn_DG)+LR_DG*Pn_DG*(Pdg_curr<LR_DG*Pn_DG)*(Pdg_curr>Pdg_min);    
                
                Edef_DC=(Eload[t]-Pdg_curr)/n_I-P_RE[t]
                Edef_DC=Edef_DC*(Edef_DC>0)
                Eb_e=(Eb_curr-Ebmin)*ef_bat
                Pdch_curr= min(min(Eb_e,Edef_DC), Pdch_max)
                
                Edef_AC=Eload[t]-Pdg_curr-min(Pinv_max, n_I*(P_RE[t]+Pdch_curr))
                Pbuy_curr=max(0, min(Edef_AC,Pbuy_max))
                Psell_curr=max(0, min(-Edef_AC,Psell_max))

            elif case5[t]:
                Edef_DC=Eload[t]/n_I-P_RE[t];
                Eb_e=(Eb[t]-Ebmin)*ef_bat;
                Pdch_curr = min(min(Eb_e,Edef_DC), Pdch_max)
                
                Edef_AC=Eload[t]-min(Pinv_max, n_I*(P_RE[t]+Pdch_curr))
                Pdg_curr=min(Edef_AC,Pn_DG)
                Pdg_curr=Pdg[t]*(Pdg_curr>=LR_DG*Pn_DG)+LR_DG*Pn_DG*(Pdg_curr<LR_DG*Pn_DG)*(Pdg_curr>Pdg_min)
                
                Pbuy_curr=max(0, min(Edef_AC-Pdg_curr,Pbuy_max))
                Psell_curr=max(0, min(Pdg_curr-Edef_AC,Psell_max))
            else:
                Edef_DC=min(Pinv_max, Eload[t]/n_I)-P_RE[t]
                Eb_e=(Eb_curr-Ebmin)*ef_bat
                Pdch_curr=min(min(Eb_e,Edef_DC)*(Edef_DC>0), Pdch_max)
                
                Edef_AC=Eload[t]-min(Pinv_max, n_I*(P_RE[t]+Pdch_curr))
                Pbuy_curr = min(Edef_AC, Pbuy_max)
                
                Pdg_curr=min(Edef_AC-Pbuy_curr,Pn_DG)
                Pdg_curr=Pdg_curr*(Pdg_curr>=LR_DG*Pn_DG)+LR_DG*Pn_DG*(Pdg_curr<LR_DG*Pn_DG)*(Pdg_curr>Pdg_min)

            Edef_DC=(Eload[t]+Psell_curr-Pdg_curr-Pbuy_curr)/n_I-(P_RE[t]+Pdch_curr-Pch_curr)
            if Edef_DC<0:
                Eb_e=(Ebmax-Eb_curr)/ef_bat
                Pch_curr=min(Eb_e, Pch_curr-Edef_DC)
                Pch_curr=min(Pch_curr,Pch_max)
    
            Esur=Eload[t]+Psell_curr-Pbuy_curr-Pdg_curr-min(Pinv_max, (P_RE[t]+Pdch_curr-Pch_curr)*n_I)
            Ens_curr=Esur*(Esur>0)
            Edump_curr = -Esur*(Esur<0)

        Ech_curr = Pch_curr * dt
        Edch_curr = Pdch_curr * dt 

        Pch.append(Pch_curr)
        Pdch.append(Pdch_curr)
        Ech.append(Ech_curr)
        Edch.append(Edch_curr)
        Pdg.append(Pdg_curr)
        Edump.append(Edump_curr)
        Ens.append(Ens_curr)
        Psell.append(Psell_curr)
        Pbuy.append(Pbuy_curr)
        Pinv.append(Pinv_curr)

        t += 1

    # for t in range(NT):
        
    
        
    #     Pdch_max, Pch_max = battery_model(Nbat, Eb[t], alfa_battery, c, k, Imax, Vnom, ef_bat) # kW    
        
    #     #%%
    #     if P_RE[t]>=(Eload[t]/n_I) and (Eload[t]<=Pinv_max):  #if PV+Pwt greater than load  (battery should charge)
            
    #         #Battery charge power calculated based on surEloadus energy and battery empty  capacity
    #         Eb_e=(Ebmax-Eb[t])/ef_bat;
    #         Pch[t]=min(Eb_e, P_RE[t]-Eload[t]/n_I);
            
    #         # Battery maximum charge power limit
    #         Pch[t]=min(Pch[t],Pch_max); 
                   
    #         Psur_AC= n_I*(P_RE[t]-Pch[t]-Eload[t]); #surplus Energy
            
    #         Psell[t]=min(Psur_AC,Psell_max); 
    #         Psell[t]=min( max(0, Pinv_max-Eload[t]),Psell[t]);
            
    #         Edump[t]=P_RE[t]-Pch[t]-(Eload[t]+Psell[t])/n_I;
                   
    #         #%% if load greater than PV+Pwt 
    #     else:
              
    #         Edef_AC=Eload[t]-min(Pinv_max, n_I*P_RE[t]);
                   
    #         price_dg=cc_gen+a*C_fuel;# DG cost ($/kWh)
    
    #         if (Cbuy[t]<= price_dg) and (price_dg<=Cbw): # Grid, DG , Bat : 1
    #             print(21)
    #             Pbuy[t]=min(Edef_AC,Pbuy_max);
    
    #             Pdg[t]=min(Edef_AC-Pbuy[t],Pn_DG);
    #             Pdg[t]=Pdg[t]*(Pdg[t]>=LR_DG*Pn_DG)+LR_DG*Pn_DG*(Pdg[t]<LR_DG*Pn_DG)*(Pdg[t]>Pdg_min);
                
    #             Edef_AC=Eload[t]-Pdg(t)-Pbuy[t]-min(Pinv_max, n_I*P_RE[t]);
    #             Edef_DC=Edef_AC/n_I*(Edef_AC>0);
    #             Eb_e=(Eb[t]-Ebmin)*ef_bat;
    #             Pdch[t]= min( Eb_e,Edef_DC);
    #             Pdch[t]= min(Pdch[t],Pdch_max);
                
    #             Esur_AC=-Edef_AC*(Edef_AC<0);
    #             Pbuy[t]=Pbuy[t]-Esur_AC*(Grid==1);
    
    
    #         elif (Cbuy[t]<= Cbw) and (Cbw<price_dg):  #Grid, Bat , DG : 2
    #             print(22)
    #             Pbuy[t]=min(Edef_AC,Pbuy_max);
               
    #             Edef_DC=(Eload[t]-Pbuy[t])/n_I-P_RE(t);
    #             Eb_e=(Eb[t]-Ebmin)*ef_bat;
    #             Pdch[t]= min( Eb_e,Edef_DC);
    #             Pdch[t]=min(Pdch[t],Pdch_max);
                
    #             Edef_AC=Eload[t]-Pbuy[t]-min(Pinv_max, n_I*(P_RE[t]+Pdch[t]));
    #             Pdg[t]=min(Edef_AC,Pn_DG);
    #             Pdg[t]=Pdg[t]*(Pdg[t]>=LR_DG*Pn_DG)+LR_DG*Pn_DG*(Pdg[t]<LR_DG*Pn_DG)*(Pdg[t]>Pdg_min);
    
    
    #         elif (price_dg<Cbuy[t]) and (Cbuy[t]<=Cbw): #DG, Grid , Bat :3
    #             print(23)
    #             Pdg[t]=min(Edef_AC,Pn_DG);
    #             Pdg[t]=Pdg(t)*(Pdg[t]>=LR_DG*Pn_DG)+LR_DG*Pn_DG*(Pdg[t]<LR_DG*Pn_DG)*(Pdg[t]>Pdg_min);
                
    #             Pbuy[t]=max(0,  min(Edef_AC-Pdg[t],Pbuy_max) );
    #             Psell[t]=max(0, min(Pdg[t]-Edef_AC,Psell_max) );
                
    #             Edef_DC=(Eload[t]-Pbuy(t)-Pdg[t])/n_I-P_RE[t];
    #             Edef_DC=Edef_DC*(Edef_DC>0);
    #             Eb_e=(Eb[t]-Ebmin)*ef_bat;
    #             Pdch[t]= min( Eb_e,Edef_DC);
    #             Pdch[t]=min(Pdch[t],Pdch_max);
    #         elif (price_dg<Cbw) and (Cbw<Cbuy[t]):  #DG, Bat , Grid :4
    #             print(24)  
    #             Pdg[t]=min(Edef_AC,Pn_DG);
    #             Pdg[t]=Pdg[t]*(Pdg[t]>=LR_DG*Pn_DG)+LR_DG*Pn_DG*(Pdg[t]<LR_DG*Pn_DG)*(Pdg[t]>Pdg_min);    
                
    #             Edef_DC=(Eload[t]-Pdg[t])/n_I-P_RE[t];
    #             Edef_DC=Edef_DC*(Edef_DC>0);
    #             Eb_e=(Eb[t]-Ebmin)*ef_bat;
    #             Pdch[t]= min( Eb_e,Edef_DC);  
    #             Pdch[t]= min(Pdch[t],Pdch_max);
                
    #             Edef_AC=Eload[t]-Pdg[t]-min(Pinv_max, n_I*(P_RE[t]+Pdch[t]) );
    #             Pbuy[t]=max(0,  min(Edef_AC,Pbuy_max) );
    #             Psell[t]=max(0, min(-Edef_AC,Psell_max) );
                
    #         elif (Cbw<price_dg) and (price_dg<Cbuy[t]):  #Bat ,DG, Grid :5
    #             print(25)
    #             Edef_DC=Eload[t]/n_I-P_RE[t];
    #             Eb_e=(Eb[t]-Ebmin)*ef_bat;
    #             Pdch[t]=min( Eb_e,Edef_DC);  
    #             Pdch[t]=min(Pdch[t],Pdch_max);  
                
    #             Edef_AC=Eload[t]-min(Pinv_max, n_I*(P_RE[t]+Pdch[t]) );
    #             Pdg[t]=min(Edef_AC,Pn_DG);
    #             Pdg[t]=Pdg[t]*(Pdg[t]>=LR_DG*Pn_DG)+LR_DG*Pn_DG*(Pdg[t]<LR_DG*Pn_DG)*(Pdg[t]>Pdg_min); 
                
    #             Pbuy[t]=max(0,  min(Edef_AC-Pdg[t],Pbuy_max) );
    #             Psell[t]=max(0, min(Pdg[t]-Edef_AC,Psell_max) );
    #         else: #Bat , Grid , DG: 6
    #             Edef_DC=min(Pinv_max, Eload[t]/n_I)-P_RE[t];
    #             Eb_e=(Eb[t]-Ebmin)*ef_bat;
    #             Pdch[t]=min( Eb_e,Edef_DC)*(Edef_DC>0);  
    #             Pdch[t]=min( Pdch[t],Pdch_max ); 
                
    #             Edef_AC=Eload[t]-min(Pinv_max, n_I*(P_RE[t]+Pdch[t]) );
    #             Pbuy[t]= min( Edef_AC, Pbuy_max);
                
    #             Pdg[t]=min(Edef_AC-Pbuy[t],Pn_DG);
    #             Pdg[t]=Pdg[t]*(Pdg[t]>=LR_DG*Pn_DG)+LR_DG*Pn_DG*(Pdg[t]<LR_DG*Pn_DG)*(Pdg[t]>Pdg_min); 
                
       
            
    #         Edef_DC=(Eload[t]+Psell[t]-Pdg[t]-Pbuy[t])/n_I-(P_RE[t]+Pdch[t]-Pch[t]);
    #         if Edef_DC<0:
    #             Eb_e=(Ebmax-Eb[t])/ef_bat;
    #             Pch[t]=min(Eb_e, Pch[t]-Edef_DC);
    #             Pch[t]=min(Pch[t],Pch_max);
    
    #         Esur=Eload[t]+Psell[t]-Pbuy[t]-Pdg[t]-min(Pinv_max, (P_RE[t]+Pdch[t]-Pch[t])*n_I);  
    #         Ens[t]=Esur*(Esur>0);
    #         Edump[t]=-Esur*(Esur<0);
    
        
    #     #%% Battery charging and discharging energy is determined based on charging and discharging power and the battery charge level is updated.
    #     Ech[t]=Pch[t]*dt;
    #     Edch[t]=Pdch[t]*dt;
    #     # index out of bounds error check
    #     Eb[t+1]=(1-self_discharge_rate)*Eb[t]+ef_bat*Ech[t]-Edch[t]/ef_bat;
    
    # print(process_time()-start)
    return np.array(Eb), np.array(Pdg), np.array(Edump), np.array(Ens), np.array(Pch), np.array(Pdch), np.array(Pbuy), np.array(Psell), np.array(Pinv)







 