import numpy as np

from Battery_Model import battery_model

"""
Energy management 
"""
def energy_management(
    Ppv, Pwt, Eload, Cn_B, Nbat, Pn_DG, NT, Pinv_max, cc_gen, Cbw, inputs
):
    Eb=np.zeros((NT, ))
    Pch=np.zeros((NT, ))
    Pdch=np.zeros((NT, ))
    Ech=np.zeros((NT, ))
    Edch=np.zeros((NT, ))
    Pdg=np.zeros((NT, ))
    Edump=np.zeros((NT, ))
    Ens=np.zeros((NT, ))
    Psell=np.zeros((NT, ))
    Pbuy=np.zeros((NT, ))
    Pinv=np.zeros((NT, ))
    Ebmax=inputs.SOC_max*Cn_B
    Ebmin=inputs.SOC_min*Cn_B
    Eb[0]=inputs.SOC_initial*Cn_B
    dt=1

    if inputs.Grid == 0:
        inputs.Pbuy_max=0
        inputs.Psell_max=0
    
    P_RE = Ppv + Pwt
    Pdg_min = 0.05*Pn_DG # LR_DG

    for t in range(NT):
        Pch_max, Pdch_max = battery_model(Nbat, Eb[t], inputs.alfa_battery, inputs.c, inputs.k, inputs.Imax, inputs.Vnom, inputs.ef_bat) # kW

        #  if PV+Pwt greater than load  (battery should charge)
        if P_RE[t] >= (Eload[t] / inputs.n_I) and Eload[t] <= Pinv_max: 
            # Battery charge power calculated based on surEloadus energy and battery empty  capacity
            Eb_e = (Ebmax - Eb[t]) / inputs.ef_bat
            Pch[t] = min(Eb_e, P_RE[t]-Eload[t]/inputs.n_I)
            
            # Battery maximum charge power limit
            Pch[t] = min(Pch[t], Pch_max)

            Psur_AC = inputs.n_I * (P_RE[t]-Pch[t]-Eload[t]) # surplus energy

            Psell[t] = min(Psur_AC, inputs.Psell_max)
            Psell[t] = min(max(0, Pinv_max-Eload[t]), Psell[t])

            Edump[t] = P_RE[t] - Pch[t] - (Eload[t] + Psell[t]) / inputs.n_I
        
        # if load greater than PV+Pwt 
        else: 
            Edef_AC = Eload[t]-min(Pinv_max, inputs.n_I * P_RE[t])
            price_dg = cc_gen + inputs.a * inputs.C_fuel # DG cost ($/kWh)

            if (inputs.Cbuy[t] <= price_dg) and (price_dg <= Cbw): # Grid, DG , Bat : 1

                Pbuy[t] = min(Edef_AC, inputs.Pbuy_max)

                Pdg[t] = min(Edef_AC-Pbuy[t], Pn_DG)
                Pdg[t] = Pdg[t] * (Pdg[t] >= inputs.LR_DG*Pn_DG) + inputs.LR_DG*Pn_DG*(Pdg[t] < inputs.LR_DG*Pn_DG) * (Pdg[t] > Pdg_min) 

                Edef_AC=Eload[t]-Pdg[t]-Pbuy[t]-min(Pinv_max, inputs.n_I*P_RE[t])
                Edef_DC=Edef_AC/inputs.n_I*(Edef_AC>0)
                Eb_e=(Eb[t]-Ebmin)*inputs.ef_bat
                Pdch[t] = min(Eb_e, Edef_DC)
                Pdch[t] = min(Pdch[t],Pdch_max)
                
                Esur_AC=-Edef_AC*(Edef_AC<0) 
                Pbuy[t]=Pbuy[t]-Esur_AC*(inputs.Grid==1)

            elif (inputs.Cbuy[t]<= Cbw) and (Cbw<price_dg): # Grid, Bat , DG : 2

                Pbuy[t]=min(Edef_AC,inputs.Pbuy_max)
                        
                Edef_DC=(Eload[t]-Pbuy[t])/inputs.n_I-P_RE[t]
                Eb_e=(Eb[t]-Ebmin)*inputs.ef_bat
                Pdch[t]= min(Eb_e,Edef_DC)
                Pdch[t]=min(Pdch[t],Pdch_max)
                
                Edef_AC=Eload[t]-Pbuy[t]-min(Pinv_max, inputs.n_I*(P_RE[t]+Pdch[t]))
                Pdg[t]=min(Edef_AC,Pn_DG)
                Pdg[t]=Pdg[t]*(Pdg[t]>=inputs.LR_DG*Pn_DG)+inputs.LR_DG*Pn_DG*(Pdg[t]<inputs.LR_DG*Pn_DG)*(Pdg[t]>Pdg_min)  
            
            elif (price_dg < inputs.Cbuy[t]) and (inputs.Cbuy[t]<=Cbw):  # DG, Grid , Bat :3
                Pdg[t]=min(Edef_AC,Pn_DG)
                Pdg[t]=Pdg[t]*(Pdg[t]>=inputs.LR_DG*Pn_DG)+inputs.LR_DG*Pn_DG*(Pdg[t]<inputs.LR_DG*Pn_DG)*(Pdg[t]>Pdg_min)
                
                Pbuy[t]=max(0, min(Edef_AC-Pdg[t],inputs.Pbuy_max))
                Psell[t]=max(0, min(Pdg[t]-Edef_AC,inputs.Psell_max))
                
                Edef_DC=(Eload[t]-Pbuy[t]-Pdg[t])/inputs.n_I-P_RE[t]
                Edef_DC=Edef_DC*(Edef_DC>0) 
                Eb_e=(Eb[t]-Ebmin)*inputs.ef_bat
                Pdch[t] = min(Eb_e,Edef_DC)
                Pdch[t]=min(Pdch[t],Pdch_max)

            elif (price_dg<Cbw) and (Cbw < inputs.Cbuy[t]):  # DG, Bat , Grid :4
                Pdg[t]=min(Edef_AC,Pn_DG)
                Pdg[t]=Pdg[t]*(Pdg[t]>=inputs.LR_DG*Pn_DG)+inputs.LR_DG*Pn_DG*(Pdg[t]<inputs.LR_DG*Pn_DG)*(Pdg[t]>Pdg_min)
                
                Edef_DC=(Eload[t]-Pdg[t])/inputs.n_I-P_RE[t]
                Edef_DC=Edef_DC*(Edef_DC>0) 
                Eb_e=(Eb[t]-Ebmin)*inputs.ef_bat
                Pdch[t]= min(Eb_e,Edef_DC)
                Pdch[t]= min(Pdch[t],Pdch_max)

                Edef_AC=Eload[t]-Pdg[t]-min(Pinv_max, inputs.n_I*(P_RE[t]+Pdch[t]))
                Pbuy[t]=max(0,  min(Edef_AC,inputs.Pbuy_max))
                Psell[t]=max(0, min(-Edef_AC,inputs.Psell_max))
                
            elif (Cbw<price_dg) and (price_dg < inputs.Cbuy[t]):  # Bat ,DG, Grid :5
                Edef_DC=Eload[t]/inputs.n_I-P_RE[t]
                Eb_e=(Eb[t]-Ebmin)*inputs.ef_bat
                Pdch[t]=min(Eb_e,Edef_DC)
                Pdch[t]=min(Pdch[t],Pdch_max)
                
                Edef_AC=Eload[t]-min(Pinv_max, inputs.n_I*(P_RE[t]+Pdch[t]))
                Pdg[t]=min(Edef_AC,Pn_DG)
                Pdg[t]=Pdg[t]*(Pdg[t]>=inputs.LR_DG*Pn_DG)+inputs.LR_DG*Pn_DG*(Pdg[t]<inputs.LR_DG*Pn_DG)*(Pdg[t]>Pdg_min)
                
                Pbuy[t]=max(0, min(Edef_AC-Pdg[t],inputs.Pbuy_max))
                Psell[t]=max(0, min(Pdg[t]-Edef_AC,inputs.Psell_max))
            else: # Bat , Grid , DG: 6
                
                Edef_DC=min(Pinv_max, Eload[t]/inputs.n_I)-P_RE[t]
                Eb_e=(Eb[t]-Ebmin)*inputs.ef_bat
                Pdch[t]=min(Eb_e,Edef_DC)*(Edef_DC>0)
                Pdch[t]=min(Pdch[t],Pdch_max) 
                
                Edef_AC=Eload[t]-min(Pinv_max, inputs.n_I*(P_RE[t]+Pdch[t]))
                Pbuy[t]= min(Edef_AC, inputs.Pbuy_max)
                
                Pdg[t]=min(Edef_AC-Pbuy[t],Pn_DG)
                Pdg[t]=Pdg[t]*(Pdg[t]>=inputs.LR_DG*Pn_DG)+inputs.LR_DG*Pn_DG*(Pdg[t]<inputs.LR_DG*Pn_DG)*(Pdg[t]>Pdg_min)
            
            Edef_DC=(Eload[t]+Psell[t]-Pdg[t]-Pbuy[t])/inputs.n_I-(P_RE[t]+Pdch[t]-Pch[t])

            if Edef_DC<0:
                Eb_e=(Ebmax-Eb[t])/inputs.ef_bat
                Pch[t]=min(Eb_e, Pch[t]-Edef_DC)
                Pch[t]=min(Pch[t],Pch_max)
            

            Esur=Eload[t]+Psell[t]-Pbuy[t]-Pdg[t]-min(Pinv_max, (P_RE[t]+Pdch[t]-Pch[t])*inputs.n_I) 
            Ens[t]=Esur*(Esur>0) 
            Edump[t]=-Esur*(Esur<0)

        # Battery charging and discharging energy is determined based on charging and discharging power and the battery charge level is updated.
        Ech[t]=Pch[t]*dt
        Edch[t]=Pdch[t]*dt

        # index out of bounds error check
        if t < NT - 1:
            Eb[t+1]=(1-inputs.self_discharge_rate)*Eb[t]+inputs.ef_bat*Ech[t]-Edch[t]/inputs.ef_bat
    
    return Eb, Pdg, Edump, Ens, Pch, Pdch, Pbuy, Psell, Pinv
 