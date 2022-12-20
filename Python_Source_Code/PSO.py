import numpy as np
import pandas as pd
from Input_Data import parameter
from Fitness import fitness
from Models import Solution, Particle
# from tictoc import tic, toc

# from EMS import energy_management
# from Battery_Model import battery_model

# from Models import Solution, Particle

# %% Loading Data 
global Eload,G, T, Vw
path='Data.csv'
Data = pd.read_csv(path, header=None).values
Eload = Data[:,0]
G = Data[:,1]
T = Data[:,2]
Vw = Data[:,3]
ins_parameter = parameter(Eload, G, T, Vw) # load input data

# %% Problem Definition

CostFunction=fitness;             # Cost Function
nVar = 5                          # number of decision variables
VarSize = (1, nVar)               # size of decision variables matrix

# Variable: PV number, WT number, Battery number, number of DG, Rated Power Inverter
VarMin = np.array([0,0,0,0,0]) # Lower bound of variables
VarMax = np.array([100,100,60,10,20]) # Upper bound of variables

VarMin = VarMin * [ins_parameter.PV,
                   ins_parameter.WT,
                   ins_parameter.Bat,
                   ins_parameter.DG, 1]
VarMax = VarMax * [ins_parameter.PV,
                   ins_parameter.WT,
                   ins_parameter.Bat,
                   ins_parameter.DG, 1]



# %% PSO Parameters

MaxIt = 100      # Max number of iterations
nPop = 50        # Population size (swarm size)
w = 1            # Inertia weight
wdamp = 0.99     # Inertia weight damping ratio
c1 = 2           # Personal learning coefficient
c2 = 2           # Global learning coefficient
# Velocity limits
VelMax = 0.3 * (VarMax - VarMin)
VelMin = -VelMax

Run_Time = 1

solution_particle = Solution()



# %%


FinalBest = {
    "Cost": float('inf'),
    "Position": None
}

for tt in range(Run_Time):
    w = 1 # intertia weight 

    # initialization
    empty_particle = Particle()
    particle = [empty_particle for _ in range(nPop)]
    particle = np.array(particle)

    GlobalBest = {
        "Cost": float('inf'),
        "Position": None
    }

    for i in range(nPop):
        # initialize position
        
        position_array = []
        for var in range(len(VarMin)):
            position_array.append(np.random.uniform(VarMin[var], VarMax[var]))
        particle[i].Position = np.array(position_array)
        
        # initialize velocity
        particle[i].Velocity = np.zeros(VarSize)
        
        # evaluation
        particle[i].Cost = fitness(particle[i].Position, Eload, G, T, Vw, ins_parameter)  
        # update personal best
        particle[i].BestPosition = particle[i].Position
        particle[i].BestCost = particle[i].Cost

        # Update global best
        if particle[i].BestCost < GlobalBest["Cost"]:
            GlobalBest["Cost"] = particle[i].BestCost
            GlobalBest["Position"] = particle[i].BestPosition


    BestCost = np.zeros((MaxIt, 1))
    MeanCost = np.zeros((MaxIt, 1))
    
   

    # PSO main loop
    for it in range(MaxIt):
        for i in range(nPop):

            # update velocity
            particle[i].Velocity = w * particle[i].Velocity + c1 * np.random.uniform(0,1,(VarSize)) * (particle[i].BestPosition - particle[i].Position) + c2 * np.random.uniform(0,1,(VarSize)) * (GlobalBest["Position"]-particle[i].Position)

            # apply velocity limits
            particle[i].Velocity = np.maximum(particle[i].Velocity, VelMin)
            particle[i].Velocity = np.minimum(particle[i].Velocity, VelMax)

            # update position
            particle[i].Position = particle[i].Position + particle[i].Velocity

            # Velocity Mirror Effect
            # TODO: double check this condition is correct
            if np.any(np.less(particle[i].Position, VarMin) | np.greater(particle[i].Position, VarMax)):
                particle[i].Velocity = -particle[i].Velocity 

            # Apply position limits
            particle[i].Position = np.maximum(particle[i].Position, VarMin)
            particle[i].Position = np.minimum(particle[i].Position, VarMax)

            # evaluation
            particle[i].Cost = fitness(particle[i].Position[0], Eload, G, T, Vw, ins_parameter)

            # update personal best
            if particle[i].Cost < particle[i].BestCost:
                particle[i].BestPosition = particle[i].Position
                particle[i].BestCost = particle[i].Cost

                # update global best
                if particle[i].BestCost < GlobalBest["Cost"]:
                    GlobalBest["Position"] = particle[i].BestPosition
                    GlobalBest["Cost"] = particle[i].BestCost
    
        BestCost[it] = GlobalBest["Cost"]
        temp = 0
        for j in range(nPop):
            temp = temp + particle[j].BestCost
        MeanCost[it] = temp / nPop

        print("Run time = ", tt)
        print("Iteration = ", it)
        print("Best Cost = ", BestCost[it])
        print("Mean Cost = ", MeanCost[it])

        w = w*wdamp

    if GlobalBest["Cost"] < FinalBest["Cost"]:
        FinalBest["Cost"] = GlobalBest["Cost"]
        FinalBest["Position"] = GlobalBest["Position"]
        FinalBest["CostCurve"] = BestCost



# %% result
from EMS import energy_management

def my_range(start,end,step):
    idx=[]
    for i in range(start,end,step):
        idx.append(i)
    return idx

inputs=ins_parameter
X=GlobalBest["Position"][0]


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

# % PV power calculation
Tc = T+(((inputs.Tnoct-20)/800)*G) # Module Temprature
Ppv = inputs.fpv*Pn_PV*(G/inputs.Gref)*(1+inputs.Tcof*(Tc-inputs.Tref)) # output power(kw)_hourly

#% Wind turbine Power Calculation
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

# %Energy management
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

RC_PV= np.zeros((inputs.n))
RC_WT= np.zeros((inputs.n))
RC_DG= np.zeros((inputs.n))
RC_B = np.zeros((inputs.n))
RC_I = np.zeros((inputs.n))
RC_CH = np.zeros((inputs.n))


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

Investment=np.zeros(inputs.n);
Investment[0]=I_Cost;
Salvage1=np.zeros(inputs.n);
Salvage1[inputs.n-1]=Salvage;
Salvage1[0]=0;
Salvage=Salvage1
Operating=np.zeros(inputs.n);
Operating[0:inputs.n+1]=inputs.MO_PV*Pn_PV + inputs.MO_WT*Pn_WT+ inputs.MO_DG\
    *Pn_DG+ inputs.MO_B*Cn_B+ inputs.MO_I*Cn_I+sum(Pbuy*inputs.Cbuy)-sum(Psell*inputs.Csell) ;
Fuel=np.zeros(inputs.n);
Fuel[0:inputs.n+1]=sum(inputs.C_fuel*q);



if(len(res_idx_L_PV)>0):
    RC_PV[res_idx_L_PV]= inputs.R_PV*Pn_PV
if(len(res_idx_L_WT)>0):
    RC_WT[res_idx_L_WT]=  inputs.R_WT*Pn_WT
if(len(res_idx_L_DG)>0):
    RC_DG[res_idx_L_DG]= inputs.R_DG*Pn_DG
if(len(res_idx_L_B)>0):
    RC_B[res_idx_L_B] =  inputs.R_B*Cn_B
if(len(res_idx_L_I)>0):
    RC_I[res_idx_L_I] =  inputs.R_I*Cn_I


Replacement=RC_PV+RC_WT+RC_DG+RC_B+RC_I;

import matplotlib.pyplot as plt

Cash_Flow=np.zeros((len(Investment),5))
Cash_Flow[:,0]=-Investment
Cash_Flow[:,1]=-Operating
Cash_Flow[:,2]=Salvage
Cash_Flow[:,3]=-Fuel
Cash_Flow[:,4]=-Replacement

# =np.concatenate((-Investment.reshape(-1,1),-Operating.reshape(-1,1),
#                           Salvage.reshape(-1,1),-Fuel.reshape(-1,1),-Replacement.reshape(-1,1)),axis=1)
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
plt.show()

plt.figure()
plt.plot(Eload-Ens,'b-.')
plt.plot(Pdg,'r')
plt.plot(Pch-Pdch,'g')
plt.plot(Ppv+Pwt,'--')
plt.legend(['Load-Ens','Pdg','Pbat','P_{RE}'])
plt.show()

plt.figure()
plt.plot(Eb/Cn_B)
plt.title('State of Charge')
plt.ylabel('SOC')
plt.xlabel('t[hour]')
plt.show()

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

plt.show()


