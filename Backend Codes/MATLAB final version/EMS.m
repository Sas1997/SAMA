%% Energy Management
function [Eb,Pdg,Edump,Ens,Pch,Pdch,Pbuy,Psell,Pinv]=...
    EMS(Ppv,Pwt,Eload,Cn_B,Nbat,Pn_DG,NT,SOC_max,SOC_min,SOC_initial,n_I,Grid,Cbuy,a,Pinv_max,LR_DG,C_fuel,Pbuy_max,Psell_max,cc_gen,Cbw,self_discharge_rate,alfa_battery,c,k,Imax,Vnom,ef_bat)

%^^^^^^^^^^^^^^READ INPUTS^^^^^^^^^^^^^^^^^^
Eb=zeros(1,NT);
Pch=zeros(1,NT);
Pdch=zeros(1,NT);
Ech=zeros(1,NT);
Edch=zeros(1,NT);
Pdg=zeros(1,NT);
Edump=zeros(1,NT);
Ens=zeros(1,NT);
Psell=zeros(1,NT);
Pbuy=zeros(1,NT);
Pinv=zeros(1,NT);
Ebmax=SOC_max*Cn_B;
Ebmin=SOC_min*Cn_B;
Eb(1)=SOC_initial*Cn_B;
dt=1;

if Grid==0
    Pbuy_max=0;
    Psell_max=0;
end

%%

P_RE=Ppv+Pwt;
if sum(Ppv+Pwt+Pbuy)==0
Pdg_min=0.25*Pn_DG; % LR_DG
else
Pdg_min=0;
end
    
Eload_max=2*max(Eload);

  for t=1:NT
    
      [Pch_max,Pdch_max]=Battery_Model(Cn_B,Nbat,Eb(t),alfa_battery,c,k,Imax,Vnom,ef_bat); % kW
      
    %%
    if P_RE(t)>=(Eload(t)/n_I) && (Eload(t)<=Eload_max) %  if PV+Pwt greater than load  (battery should charge)
        
        % Battery charge power calculated based on surEloadus energy and battery empty  capacity
        Eb_e=(Ebmax-Eb(t))/sqrt(ef_bat);
        Pch(t)=min(Eb_e, P_RE(t)-Eload(t)/n_I);
        
        % Battery maximum charge power limit
        Pch(t)=min(Pch(t),Pch_max); % Pch<=Pch_max
               
        Psur_AC= n_I*(P_RE(t)-Pch(t))-Eload(t); % surplus Energy
        
        Psell(t)=min(Psur_AC,Psell_max); % Psell<=Psell_max
        Psell(t)=min( max(0, Pinv_max-Eload(t)),Psell(t));
        
        Edump(t)=P_RE(t)-Pch(t)-(Eload(t)+Psell(t))/n_I;
               
        %% if load greater than PV+Pwt 
    else
        
        Edef_AC=Eload(t)-min(Pinv_max, n_I*P_RE(t));
               
        price_dg=cc_gen+a*C_fuel;%% DG cost ($/kWh)

        if (Cbuy(t)<= price_dg) && (price_dg<=Cbw) % Grid, DG , Bat : 1
            
            Pbuy(t)=min(Edef_AC,Pbuy_max);
            
            Pdg(t)=min(Edef_AC-Pbuy(t),Pn_DG);
            Pdg(t)=Pdg(t)*(Pdg(t)>=LR_DG*Pn_DG)+LR_DG*Pn_DG*(Pdg(t)<LR_DG*Pn_DG)*(Pdg(t)>Pdg_min);
            
            Edef_AC=Eload(t)-Pdg(t)-Pbuy(t)-min(Pinv_max, n_I*P_RE(t));
            Edef_DC=(Edef_AC/n_I)*(Edef_AC>0);
            Eb_e=(Eb(t)-Ebmin)*sqrt(ef_bat);
            Pdch(t)= min( Eb_e,Edef_DC);
            Pdch(t)= min(Pdch(t),Pdch_max);
            
            Esur_AC=-Edef_AC*(Edef_AC<0);
            Pbuy(t)=Pbuy(t)-Esur_AC*(Grid==1);
            
        elseif (Cbuy(t)<= Cbw) && (Cbw<price_dg)  % Grid, Bat , DG : 2
            
            Pbuy(t)=min(Edef_AC,Pbuy_max);
                       
            Edef_DC=(Eload(t)-Pbuy(t))/n_I-P_RE(t);
            Eb_e=(Eb(t)-Ebmin)*sqrt(ef_bat);
            Pdch(t)= min( Eb_e,Edef_DC);
            Pdch(t)=min(Pdch(t),Pdch_max);
            
            Edef_AC=Eload(t)-Pbuy(t)-min(Pinv_max, n_I*(P_RE(t)+Pdch(t)));
            Pdg(t)=min(Edef_AC,Pn_DG);
            Pdg(t)=Pdg(t)*(Pdg(t)>=LR_DG*Pn_DG)+LR_DG*Pn_DG*(Pdg(t)<LR_DG*Pn_DG)*(Pdg(t)>Pdg_min);
            
        elseif (price_dg<Cbuy(t)) && (Cbuy(t)<=Cbw)  % DG, Grid , Bat :3
            Pdg(t)=min(Edef_AC,Pn_DG);
            Pdg(t)=Pdg(t)*(Pdg(t)>=LR_DG*Pn_DG)+LR_DG*Pn_DG*(Pdg(t)<LR_DG*Pn_DG)*(Pdg(t)>Pdg_min);
            
            Pbuy(t)=max(0,  min(Edef_AC-Pdg(t),Pbuy_max) );
            Psell(t)=max(0, min(Pdg(t)-Edef_AC,Psell_max) );
            
            Edef_DC=(Eload(t)-Pbuy(t)-Pdg(t))/n_I-P_RE(t);
            Edef_DC=Edef_DC*(Edef_DC>0);
            Eb_e=(Eb(t)-Ebmin)*sqrt(ef_bat);
            Pdch(t)= min( Eb_e,Edef_DC);
            Pdch(t)=min(Pdch(t),Pdch_max);
            
        elseif (price_dg<Cbw) && (Cbw<Cbuy(t))  % DG, Bat , Grid :4
            Pdg(t)=min(Edef_AC,Pn_DG);
            Pdg(t)=Pdg(t)*(Pdg(t)>=LR_DG*Pn_DG)+LR_DG*Pn_DG*(Pdg(t)<LR_DG*Pn_DG)*(Pdg(t)>Pdg_min);    
            
            Edef_DC=(Eload(t)-Pdg(t))/n_I-P_RE(t);
            Edef_DC=Edef_DC*(Edef_DC>0);
            Eb_e=(Eb(t)-Ebmin)*sqrt(ef_bat);
            Pdch(t)= min( Eb_e,Edef_DC);  
            Pdch(t)= min(Pdch(t),Pdch_max);

            Edef_AC=Eload(t)-Pdg(t)-min(Pinv_max, n_I*(P_RE(t)+Pdch(t)) );
            Pbuy(t)=max(0,  min(Edef_AC,Pbuy_max) );
            Psell(t)=max(0, min(-Edef_AC,Psell_max) );
            
            elseif (Cbw<price_dg) && (price_dg<Cbuy(t))  % Bat ,DG, Grid :5
            Edef_DC=Eload(t)/n_I-P_RE(t);
            Eb_e=(Eb(t)-Ebmin)*sqrt(ef_bat);
            Pdch(t)=min( Eb_e,Edef_DC);  
            Pdch(t)=min(Pdch(t),Pdch_max);  
            
            Edef_AC=Eload(t)-min(Pinv_max, n_I*(P_RE(t)+Pdch(t)) );
            Pdg(t)=min(Edef_AC,Pn_DG);
            Pdg(t)=Pdg(t)*(Pdg(t)>=LR_DG*Pn_DG)+LR_DG*Pn_DG*(Pdg(t)<LR_DG*Pn_DG)*(Pdg(t)>Pdg_min); 
            
            Pbuy(t)=max(0,  min(Edef_AC-Pdg(t),Pbuy_max) );
            Psell(t)=max(0, min(Pdg(t)-Edef_AC,Psell_max) );
            
        else % Bat , Grid , DG: 6
            
            Edef_DC=min(Pinv_max, Eload(t)/n_I)-P_RE(t);
            Eb_e=(Eb(t)-Ebmin)*sqrt(ef_bat);
            Pdch(t)=min( Eb_e,Edef_DC)*(Edef_DC>0);  
            Pdch(t)=min( Pdch(t),Pdch_max ); 
            
            Edef_AC=Eload(t)-min(Pinv_max, n_I*(P_RE(t)+Pdch(t)) );
            Pbuy(t)= min( Edef_AC, Pbuy_max);
            
            Pdg(t)=min(Edef_AC-Pbuy(t),Pn_DG);
            Pdg(t)=Pdg(t)*(Pdg(t)>=LR_DG*Pn_DG)+LR_DG*Pn_DG*(Pdg(t)<LR_DG*Pn_DG)*(Pdg(t)>Pdg_min); 
            
        end
        
        Edef_DC=(Eload(t)+Psell(t)-Pdg(t)-Pbuy(t))/n_I-(P_RE(t)+Pdch(t)-Pch(t));

        if Edef_DC<0
        Eb_e=(Ebmax-Eb(t))/sqrt(ef_bat);
        Pch(t)=min(Eb_e, Pch(t)-Edef_DC);
        Pch(t)=min(Pch(t),Pch_max);
        end

        Esur=Eload(t)+Psell(t)-Pbuy(t)-Pdg(t)-min(Pinv_max, (P_RE(t)+Pdch(t)-Pch(t))*n_I);  
        Ens(t)=Esur*(Esur>0);
        Edump(t)=-Esur*(Esur<0);
    end
    
    %% Battery charging and discharging energy is determined based on charging and discharging power and the battery charge level is updated.
    Ech(t)=Pch(t)*dt;
    Edch(t)=Pdch(t)*dt;
    Eb(t+1)=(1-self_discharge_rate)*Eb(t)+sqrt(ef_bat)*Ech(t)-Edch(t)/sqrt(ef_bat);
            
    
  end

end