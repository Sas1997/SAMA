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
    Eb=np.zeros(NT+1);
    Pch=np.zeros(NT);
    Pdch=np.zeros(NT);
    Ech=np.zeros(NT);
    Edch=np.zeros(NT);
    Pdg=np.zeros(NT);
    Ens=np.zeros(NT);
    Psell=np.zeros(NT);
    Pbuy=np.zeros(NT);
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
    for t in range(NT):
        
    
        
        Pdch_max, Pch_max = battery_model(Nbat, Eb[t], alfa_battery, c, k, Imax, Vnom, ef_bat) # kW    
        
        #%%
        if P_RE[t]>=(Eload[t]/n_I) and (Eload[t]<=Pinv_max):  #if PV+Pwt greater than load  (battery should charge)
            
            #Battery charge power calculated based on surEloadus energy and battery empty  capacity
            Eb_e=(Ebmax-Eb[t])/ef_bat;
            Pch[t]=min(Eb_e, P_RE[t]-Eload[t]/n_I);
            
            # Battery maximum charge power limit
            Pch[t]=min(Pch[t],Pch_max); 
                   
            Psur_AC= n_I*(P_RE[t]-Pch[t]-Eload[t]); #surplus Energy
            
            Psell[t]=min(Psur_AC,Psell_max); 
            Psell[t]=min( max(0, Pinv_max-Eload[t]),Psell[t]);
            
                   
            #%% if load greater than PV+Pwt 
        else:
              
            Edef_AC=Eload[t]-min(Pinv_max, n_I*P_RE[t]);
                   
            price_dg=cc_gen+a*C_fuel;# DG cost ($/kWh)
    
            if (Cbuy[t]<= price_dg) and (price_dg<=Cbw): # Grid, DG , Bat : 1
                print(21)
                Pbuy[t]=min(Edef_AC,Pbuy_max);
    
                Pdg[t]=min(Edef_AC-Pbuy[t],Pn_DG);
                Pdg[t]=Pdg[t]*(Pdg[t]>=LR_DG*Pn_DG)+LR_DG*Pn_DG*(Pdg[t]<LR_DG*Pn_DG)*(Pdg[t]>Pdg_min);
                
                Edef_AC=Eload[t]-Pdg(t)-Pbuy[t]-min(Pinv_max, n_I*P_RE[t]);
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
    
        
        #%% Battery charging and discharging energy is determined based on charging and discharging power and the battery charge level is updated.
        Ech[t]=Pch[t]*dt;
        Edch[t]=Pdch[t]*dt;
        # index out of bounds error check
        Eb[t+1]=(1-self_discharge_rate)*Eb[t]+ef_bat*Ech[t]-Edch[t]/ef_bat;
    
    return Eb, Pdg, Ens, Pch, Pdch, Pbuy, Psell







 