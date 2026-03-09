import numpy as np
import pandas as pd
import sys
np.set_printoptions(threshold=sys.maxsize)

# Reading meteorological data and load data
def Heat_Pump_Model(T, P, Hload, Cload):

    max_heating_load_hourly = np.amax(Hload) # maximum heating load [kW]
    max_cooling_load_hourly = np.amax(Cload) # maximum cooling load [kW]

    # Calculating the indoor wet bulb temperature

    T_in_design = 22.22 # indoor design temperature [C]
    w_in_design = 0.005 # indoor design humidity ratio [kg/kg da]
    P_sat_in = 0.623692418 + 0.0424692499*T_in_design + 0.00134403923*(T_in_design)**2 + 0.0000309447379*(T_in_design)**3 + 3.74294905E-07*(T_in_design)**4
    P_in = P
    RH_in = w_in_design*P_in/(w_in_design*P_sat_in + 0.622*P_sat_in)
    RH_in_100 = RH_in*100
    T_iwb = T_in_design*np.arctan(0.151977*(RH_in_100 + 8.313659)**0.5) + np.arctan(T_in_design + RH_in_100) - np.arctan(RH_in_100 - 1.676331) + (0.00391838*(RH_in_100)**1.5)*np.arctan(0.023101*RH_in_100) - 4.686035

    # Reading the heat pump models' coefficients

    model_capacity = np.genfromtxt('C:/Users/alisa/Desktop/FAST/SAMA V1.04-HP added/content/Heat Pump/models.csv', delimiter=',', skip_header=1, dtype=str)
    heating_COP = np.genfromtxt('C:/Users/alisa/Desktop/FAST/SAMA V1.04-HP added/content/Heat Pump/heating_COP.csv', delimiter=',', skip_header=1)
    cooling_COP = np.genfromtxt('C:/Users/alisa/Desktop/FAST/SAMA V1.04-HP added/content/Heat Pump/cooling_COP.csv', delimiter=',', skip_header=1)

    # Selecting a heat pump based on the maximum demands (worst case scenario)

    BTU_convert = 3412.142 # rate we need to multiply kW demand by to get BTU/hr

    def selectHP(demand_kW):
        demand_BTU = demand_kW * BTU_convert # first, convert the demand in kW to BTU/hr
        hp = None # placeholder for selected heat pump

        for ratedCapacity, row in zip(model_capacity[:,1],model_capacity[:,0]):  # iterate through rated capacity column until we find the rated capacity that satisfies the demand
            if int(ratedCapacity) >= demand_BTU:
                hp_capacity = int(ratedCapacity)
                hp_model = row
                break
        return hp_model, hp_capacity # returns heat pump model and its rated capacity in BTU/hr

    hp_model_heating, hp_capacity_heating = selectHP(max_heating_load_hourly)
    hp_model_cooling, hp_capacity_cooling = selectHP(max_cooling_load_hourly)

    # Selecting the final HP size

    if hp_capacity_heating > hp_capacity_cooling:

        HP_size = hp_capacity_heating
        hp_model = hp_model_heating

    else:

        HP_size = hp_capacity_cooling
        hp_model = hp_model_cooling

    # Hourly COP and hourly power consumption in heating mode

    def heatingCOP_electricity (temp, heat_load):
        for ratedCapacity, b0, b1, b2, min_COP_h, max_COP_h in zip(heating_COP[:,0],heating_COP[:,4],heating_COP[:,5],heating_COP[:,6],heating_COP[:,7],heating_COP[:,8]):  # iterate through rated capacity column until we find the related coefficients for COP equation
            if int(ratedCapacity) == HP_size:
                b_0 = float(b0)
                b_1 = float(b1)
                b_2 = float(b2)
                COP = b_0 + b_1*temp + b_2*temp**2
                COP[COP > max_COP_h] = max_COP_h
                COP[COP < 1] = 1
                kWh = heat_load/COP
        return COP, min_COP_h, kWh # returns heat pump's hourly COP and electricity consumption arrays

    COP_hp_heating, min_COP_heating, power_hp_heating = heatingCOP_electricity(T, Hload)


    # Hourly COP and hourly power consumption in cooling mode

    def coolingCOP_electricity(T_Inside, T_iwb, T_Outside, cooling_load):
        for ratedCapacity, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, min_COP_c, max_COP_c in zip(
                cooling_COP[:, 0], cooling_COP[:, 14], cooling_COP[:, 15], cooling_COP[:, 16], cooling_COP[:, 17],
                cooling_COP[:, 18], cooling_COP[:, 19], cooling_COP[:, 20], cooling_COP[:, 21], cooling_COP[:, 22],
                cooling_COP[:, 23], cooling_COP[:, 24], cooling_COP[:, 25], cooling_COP[:, 26], cooling_COP[:, 27],
                cooling_COP[:,
                28]):  # iterate through rated capacity column until we find the related coefficients for COP equation
            if int(ratedCapacity) == HP_size:
                b_0 = float(b0)
                b_1 = float(b1)
                b_2 = float(b2)
                b_3 = float(b3)
                b_4 = float(b4)
                b_5 = float(b5)
                b_6 = float(b6)
                b_7 = float(b7)
                b_8 = float(b8)
                b_9 = float(b9)
                b_10 = float(b10)
                b_11 = float(b11)
                b_12 = float(b12)
                COP = b_0 + b_1 * T_Inside + b_2 * T_iwb + b_3 * T_Outside + b_4 * T_Inside * T_iwb + b_5 * T_Inside * T_Outside + b_6 * T_iwb * T_Outside + b_7 * T_Inside ** 2 + b_8 * T_iwb ** 2 + b_9 * T_Outside ** 2 + b_10 * (
                            T_Inside ** 2) * T_Outside + b_11 * (
                                  T_iwb ** 2) * T_Outside + b_12 * T_Inside * T_iwb * T_Outside
                COP[COP > max_COP_c] = max_COP_c
                COP[COP < 1] = 1
                kWh = cooling_load / COP
        return COP, min_COP_c, kWh  # returns heat pump's hourly COP and electricity consumption arrays

    COP_hp_cooling, min_COP_cooling, power_hp_cooling = coolingCOP_electricity(T_in_design, T_iwb, T, Cload)

    power_hp_total = power_hp_heating + power_hp_cooling
    return power_hp_total, power_hp_heating, power_hp_cooling, COP_hp_heating, COP_hp_cooling, hp_model, HP_size