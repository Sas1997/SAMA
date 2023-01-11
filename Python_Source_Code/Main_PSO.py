import pandas as pd
from Input_Data import *
from Fitness import CostFunction
import numpy as np
from copy import copy, deepcopy

# %% first class
class Solution():
    def __init__(self):
        self.BestCost = []
        self.BestSol = []
        self.CostCurve = []

class Particle():
    def __init__(self):
        self.Position = []
        self.Cost = []
        self.Velocity = []
        self.Best=None


def shpere(x, Eload, G, T, Vw,ins_parameter):
    z=np.array(x)
    z=np.sum(x**2)
    return z

# %% Loading Data 
path='Data.csv'
Data = pd.read_csv(path, header=None).values
Eload = np.array(Data[:,0])
G = np.array(Data[:,1])
T = np.array(Data[:,2])
Vw = np.array(Data[:,3])


# %% Problem Definition
nVar = 5                          # number of decision variables
VarSize = (1, nVar)               # size of decision variables matrix

# Variable: PV number, WT number, Battery number, number of DG, Rated Power Inverter
VarMin = np.array([0,0,0,0,0]) # Lower bound of variables
VarMax = np.array([100,100,60,10,20]) # Upper bound of variables

VarMin = VarMin * np.array([PV,WT,Bat,DG, 1])
VarMax = VarMax * np.array([PV,WT,Bat,DG, 1])
Cbuy = calcTouCbuy(daysInMonth,months, holidays)

# %% PSO Parameters
MaxIt = 100     # Max number of iterations
nPop = 50        # Population size (swarm size)
w = 1            # Inertia weight
wdamp = 0.99     # Inertia weight damping ratio
c1 = 2           # Personal learning coefficient
c2 = 2           # Global learning coefficient
# Velocity limits
VelMax = 0.3 * (VarMax - VarMin)
VelMin = -VelMax
Run_Time = 1
Sol= np.array([Solution() for _ in range(Run_Time)])

for tt in range(Run_Time):
    Position = np.random.uniform(VarMin, VarMax, (1, nPop, nVar))[0]
    Cost = CostFunctionMatrix(Position, Eload, G, T, Vw)
    # Cost = np.full((nPop,), CostFunction(particle[i].Position,Eload, G, T, Vw))
    Velocity = np.zeros(VarSize)
    Best=None

    particle = np.full((nPop,), Particle())
    # particle = np.array([Particle() for _ in range(nPop)])
    GlobalBest=Particle()
    GlobalBest.Cost=np.inf
    
    # for i in range(nPop):
    #     #Initialize Position
    #     particle[i].Position=np.random.uniform(VarMin, VarMax,VarSize)[0]
        
    #     #Initialize Velocity
    #     particle[i].Velocity=np.zeros(VarSize)
    
    #     #Evaluation
    #     particle[i].Cost= CostFunction(particle[i].Position,Eload, G, T, Vw)
    #     #Update Personal Best
    #     particle[i].Best=deepcopy(particle[i])

    #     #Update Global Best
    #     if particle[i].Best.Cost<GlobalBest.Cost:
    #         GlobalBest=particle[i].Best;   

    # BestCost = np.zeros((MaxIt, 1))
    # MeanCost = np.zeros((MaxIt, 1))
    
    
    
    
    #%% PSO Main Loop
    for it in range(MaxIt):
        
        for i in range(nPop):
    
            #Update Velocity
            particle[i].Velocity = w*particle[i].Velocity+c1*np.random.rand(VarSize[1])\
                *(particle[i].Best.Position-particle[i].Position)\
                    +c2*np.random.rand(VarSize[1])*(GlobalBest.Position-particle[i].Position)
             
            # Apply Velocity Limits
            particle[i].Velocity = np.maximum(particle[i].Velocity,VelMin);
            particle[i].Velocity = np.minimum(particle[i].Velocity,VelMax);
            
            # Update Position
            particle[i].Position = particle[i].Position + particle[i].Velocity;
            
            # Velocity Mirror Effect
            IsOutside=(np.less(particle[i].Position, VarMin)\
                        | np.greater(particle[i].Position, VarMax))[0]
            IsOutside=IsOutside.reshape(1,-1)    
            particle[i].Velocity[IsOutside]=-particle[i].Velocity[IsOutside];
            
            # Apply Position Limits
            particle[i].Position = np.maximum(particle[i].Position,VarMin);
            particle[i].Position = np.minimum(particle[i].Position,VarMax);
        
            # Evaluation
            particle[i].Cost= CostFunction(particle[i].Position,Eload, G, T, Vw);
     
            # Update Personal Best
            if particle[i].Cost<particle[i].Best.Cost:
                particle[i].Best.Position=particle[i].Position;
                particle[i].Best.Cost=particle[i].Cost;
                # Update Global Best
                if particle[i].Best.Cost<GlobalBest.Cost:     
                    GlobalBest=deepcopy(particle[i].Best);
        BestCost[it] = GlobalBest.Cost
        
        temp = 0
        for j in range(nPop):
            temp = temp + particle[j].Best.Cost
        MeanCost[it] = temp / nPop
        print('Run time = '+ str(tt)+\
              ' , Iteration = '+ str(it)+\
                  ', Best Cost = '+str(np.round(BestCost[it][0],4))+ \
                      ', Mean Cost = '+ str(np.round(MeanCost[it][0],4)))
        

        w = w*wdamp

    Sol[tt].BestCost=GlobalBest.Cost;
    Sol[tt].BestSol=GlobalBest.Position;
    Sol[tt].CostCurve=BestCost;
    
#%% final result

Best=particle = [Sol[t].BestCost for t in range(len(Sol))]
index=np.argmin(Best);
X=Sol[index].BestSol;

#%% result 1
import matplotlib.pyplot as plt
plt.figure(1)
plt.plot(Sol[index].CostCurve,'-.');
plt.xlabel('iteration');
plt.ylabel('Cost of Best Solution ')
plt.title('Converage Curve')

#%% result figure

if(len(X))==1:
    X=X[0]

NT=len(Eload);        # time step numbers
Npv=round(X[0]);      # PV number
Nwt=round(X[1]);      # WT number
Nbat=round(X[2]);     # Battery pack number
N_DG=round(X[3]);     # number of Diesel Generator
Cn_I=X[4];            # Inverter Capacity

Pn_PV=Npv*Ppv_r;   # PV Total Capacity
Pn_WT=Nwt*Pwt_r;   # WT Total Capacity
Cn_B=Nbat*Cbt_r;   # Battery Total Capacity
Pn_DG=N_DG*Cdg_r;  # Diesel Total Capacity

#%% PV Power Calculation
Tc   = T+(((Tnoct-20)/800)*G); # Module Temprature
Ppv = fpv*Pn_PV*(G/Gref)*(1+Tcof*(Tc-Tref)); # output power(kw)_hourly

# %% Wind turbine Power Calculation
v1=Vw;     #hourly wind speed
v2=((h_hub/h0)**(alfa_wind_turbine))*v1; # v1 is the speed at a reference height;v2 is the speed at a hub height h2
Pwt=np.zeros(8760);


Pwt[v2<v_cut_in]=0
Pwt[v2>v_cut_out]=0
true_value=np.logical_and(v_cut_in<=v2,v2<v_rated)
Pwt[np.logical_and(v_cut_in<=v2,v2<v_rated)]=v2[true_value]**3 *(Pwt_r/(v_rated**3-v_cut_in**3))-(v_cut_in**3/(v_rated**3-v_cut_in**3))*(Pwt_r);
Pwt[np.logical_and(v_rated<=v2,v2<v_cut_out)]=Pwt_r
Pwt=Pwt*Nwt;


#%% Energy Management 
#% Battery Wear Cost
from EMS import energy_management
if Cn_B>0:
    Cbw=R_B*Cn_B/(Cn_B*Q_lifetime*np.sqrt(ef_bat) );
else:
    Cbw=0;


#  DG Fix cost
cc_gen=b*Pn_DG*C_fuel+R_DG*Pn_DG/TL_DG+MO_DG;


(Eb, Pdg, Edump, Ens, Pch, Pdch, Pbuy, Psell, Pinv) =\
    energy_management(Ppv,Pwt,Eload,Cn_B,Nbat,Pn_DG,NT,
                      SOC_max,SOC_min,SOC_initial,
                      n_I,Grid,Cbuy,a,Cn_I,LR_DG,C_fuel,Pbuy_max,Psell_max,cc_gen,Cbw,
                      self_discharge_rate,alfa_battery,c,k,Imax,Vnom,ef_bat)

q=(a*Pdg+b*Pn_DG)*(Pdg>0);   # Fuel consumption of a diesel generator 

#%% installation and operation cost

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

RC_PV[np.arange(L_PV+1,n,L_PV)]= R_PV*Pn_PV/(1+ir)**(np.arange(1.001*L_PV,n,L_PV)) ;
RC_WT[np.arange(L_WT+1,n,L_WT)]= R_WT*Pn_WT/(1+ir)** (np.arange(1.001*L_WT,n,L_WT)) ;
RC_DG[np.arange(L_DG+1,n,L_DG).astype(np.int32)]= R_DG*Pn_DG/(1+ir)**(np.arange(1.001*L_DG,n,L_DG)) ;
RC_B[np.arange(L_B+1,n,L_B)] = R_B*Cn_B /(1+ir)**(np.arange(1.001*L_B,n,L_B)) ;
RC_I[np.arange(L_I+1,n,L_I)] = R_I*Cn_I /(1+ir)**(np.arange(1.001*L_I,n,L_I)) ;
RC_CH[np.arange(L_CH+1,n,L_CH)]  = R_CH /(1+ir)**(np.arange(1.001*L_CH,n,L_CH)) ;
R_Cost=RC_PV+RC_WT+RC_DG+RC_B+RC_I+RC_CH;

#Total M&O Cost ($/year)
MO_Cost=( MO_PV*Pn_PV + MO_WT*Pn_WT+ MO_DG*np.sum(Pn_DG>0)+ \
         MO_B*Cn_B+ MO_I*Cn_I +MO_CH)/(1+ir)**np.array(range(1,n+1)) ;

# DG fuel Cost
C_Fu= sum(C_fuel*q)/(1+ir)**np.array(range(1,n+1));

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


LCOE=CRF*NPC/np.sum(Eload-Ens+Psell);                #Levelized Cost of Energy ($/kWh)
LEM=(DG_Emissions+Grid_Emissions)/sum(Eload-Ens);    #Levelized Emissions(kg/kWh)


LPSP=np.sum(Ens)/np.sum(Eload);   

RE=1-np.sum(Pdg+Pbuy)/np.sum(Eload+Psell-Ens);
if(np.isnan(RE)):
    RE=0;
#%%
Investment=np.zeros(n);
Investment[0]=I_Cost;
Salvage1=np.zeros(n);
Salvage1[n-1]=Salvage
Salvage1[0]=0;
Salvage=Salvage1
Operating=np.zeros(n)
Operating[0:n+1]=MO_PV*Pn_PV + MO_WT*Pn_WT+ MO_DG\
    *Pn_DG+ MO_B*Cn_B+ MO_I*Cn_I+sum(Pbuy*Cbuy)-sum(Psell*Csell) ;
Fuel=np.zeros(n)
Fuel[0:n+1]=sum(C_fuel*q);

#%%
RC_PV[np.arange(L_PV+1,n,L_PV)]= R_PV*Pn_PV
RC_WT[np.arange(L_WT+1,n,L_WT)]=R_WT*Pn_WT 
RC_DG[np.arange(L_DG+1,n,L_DG).astype(np.int32)]= R_DG*Pn_DG
RC_B[np.arange(L_B+1,n,L_B)] = R_B*Cn_B
RC_I[np.arange(L_I+1,n,L_I)] =R_I*Cn_I 
Replacement=RC_PV+RC_WT+RC_DG+RC_B+RC_I;




Cash_Flow=np.zeros((len(Investment),5))
Cash_Flow[:,0]=-Investment
Cash_Flow[:,1]=-Operating
Cash_Flow[:,2]=Salvage
Cash_Flow[:,3]=-Fuel
Cash_Flow[:,4]=-Replacement

plt.figure()
for kk in range(5):
    plt.bar(range(0,25),Cash_Flow[:,kk])
plt.legend(['Capital','Operating','Salvage','Fuel','Replacement'])
plt.title('Cash Flow')
plt.xlabel('Year')
plt.ylabel('$')

# ///
print( ' ')
print( 'System Size ')
print('Cpv  (kW) = ', str(Pn_PV))
print('Cwt  (kW) = ' ,str(Pn_WT))
print('Cbat (kWh) = ' ,str(Cn_B))
print('Cdg  (kW) = ' ,str(Pn_DG))
print('Cinverter (kW) = ', str(Cn_I))

print(' ')
print( 'Result: ')
print('NPC  = ', str(NPC) ,' $ ')
print('LCOE  = ', str(LCOE) ,' $/kWh ')
print('Operation Cost  = ', str(Operating_Cost), ' $ ')
print('Initial Cost  =, ' , str(I_Cost), ' $ ')
print('RE  = ', str(100*RE) ,' % ')
print('Total operation and maintainance cost  = ', str(sum(MO_Cost)), ' $ ')

print('LPSP  = ', str(100*LPSP) ,' % ')
print('excess Elecricity = ', str(sum(Edump)))


print('Total power bought from Grid= ', str(sum(Pbuy)), ' kWh ')
print('Total Money paid to the Grid= ', str(sum(Grid_Cost)), ' $ ')
print('Total Money paid by the user= ', str(np.sum(NPC)), ' $ ')
print('Grid Sales = ', str(sum(Psell)), ' kWh ')
print('LEM  = ', str(LEM), ' kg/kWh ')
print('PV Power  = ', str(sum(Ppv)),' kWh ')
print('WT Power  = ', str(sum(Pwt)), ' kWh ')
print('DG Power  = ', str(sum(Pdg)), ' kWh ')
print('total fuel consumed by DG   = ', str(sum(q)) ,' (kg/year) ')

print('DG Emissions   = ', str(DG_Emissions),' (kg/year) ')
print('Grid Emissions   = ', str(Grid_Emissions), ' (kg/year) ')


# Plot Results

plt.figure()
plt.plot(Pbuy)
plt.plot(Psell)
plt.legend(['Buy','sell'])
plt.ylabel('Pgrid (kWh)')
plt.xlabel('t(hour)')

plt.figure()
plt.plot(Eload-Ens,'b-.')
plt.plot(Pdg,'r')
plt.plot(Pch-Pdch,'g')
plt.plot(Ppv+Pwt,'--')
plt.legend(['Load-Ens','Pdg','Pbat','P_{RE}'])

plt.figure()
plt.plot(Eb/Cn_B)
plt.title('State of Charge')
plt.ylabel('SOC')
plt.xlabel('t[hour]')

# Plot results for one specific day 
Day=180;
t1=Day*24+1;
t2=Day*24+24;

plt.figure(figsize=(10,10))
plt.title(['Results for ' ,str(Day), ' -th day']) 
plt.subplot(4,4,1)
plt.plot(Eload)
plt.title('Load Profile')
plt.ylabel('E_{load} [kWh]')
plt.xlabel('t[hour]')
plt.xlim([t1,t2])



plt.subplot(4,4,5)
plt.plot(Eload)
plt.title('Load Profile')
plt.ylabel('E_{load} [kWh]')
plt.xlabel('t[hour]')
plt.xlim([t1,t2])



plt.subplot(4,4,2)
plt.plot(G)
plt.title('Plane of Array Irradiance')
plt.ylabel('G[W/m^2]')
plt.xlabel('t[hour]')
plt.xlim([t1, t2])

plt.subplot(4,4,6)
plt.plot(T)
plt.title('Ambient Temperature')
plt.ylabel('T[^o C]')
plt.xlabel('t[hour]')
plt.xlim([t1 ,t2])

plt.subplot(4,4,3)
plt.plot(Ppv)
plt.title('PV Power')
plt.ylabel('P_{pv} [kWh]')
plt.xlabel('t[hour]')
plt.xlim([t1, t2])

plt.subplot(4,4,4)
plt.plot(Ppv)
plt.title('PV Power')
plt.ylabel('P_{pv} [kWh]')
plt.xlabel('t[hour]')
plt.xlim([t1, t2])



plt.subplot(4,4,7)
plt.plot(Pwt)
plt.title('WT Energy')
plt.ylabel('P_{wt} [kWh]')
plt.xlabel('t[hour]')
plt.xlim([t1, t2])
plt.subplot(4,4,8)
plt.plot(Pwt)
plt.title('WT Energy')
plt.ylabel('P_{wt} [kWh]')
plt.xlabel('t[hour]')
plt.xlim([t1, t2])

plt.subplot(4,4,9)
plt.plot(Pdg)
plt.title('Diesel Generator Energy')
plt.ylabel('E_{DG} [kWh]')
plt.xlabel('t[hour]')
plt.xlim([t1,t2])
plt.subplot(4,4,10)
plt.plot(Pdg)
plt.title('Diesel Generator Energy')
plt.ylabel('E_{DG} [kWh]')
plt.xlabel('t[hour]')
plt.xlim([t1,t2])

plt.subplot(4,4,11)
plt.plot(Eb)
plt.title('Battery Energy Level')
plt.ylabel('E_{b} [kWh]')
plt.xlabel('t[hour]')
plt.xlim([t1, t2])

plt.subplot(4,4,12)
plt.plot(Eb/Cn_B)
plt.title('State of Charge')
plt.ylabel('SOC')
plt.xlabel('t[hour]')
plt.xlim([t1,t2])

plt.subplot(4,4,13)
plt.plot(Ens)
plt.title('Loss of Power Suply')
plt.ylabel('LPS[kWh]')
plt.xlabel('t[hour]')
plt.xlim([t1,t2])

plt.subplot(4,4,14)
plt.plot(Edump)
plt.title('Dumped Energy')
plt.ylabel('E_{dump} [kWh]')
plt.xlabel('t[hour]')
plt.xlim([t1,t2])

plt.subplot(4,4,15)
plt.bar(range(len(Pdch)),Pdch)
plt.title('Battery decharge Energy')
plt.ylabel('E_{dch} [kWh]')
plt.xlabel('t[hour]')
plt.xlim([t1, t2])

plt.subplot(4,4,16)
plt.bar(range(len(Pdch)),Pch)
plt.title('Battery charge Energy')
plt.ylabel('E_{ch} [kWh]')
plt.xlabel('t[hour]')
plt.xlim([t1,t2])



