def top_down_pricing(Total_PV_price):

    # Pricing method 1/top down

    # NREL percentages
    r_PV = 0.126
    r_inverter = 0.1171
    r_Fieldwork = 0.06537
    r_Officework = 0.2595
    r_Other = 0.2185
    r_Electrical_BoS = 0.1242
    r_Structrual_BoS = 0.08837


    # Engineering Costs (Per/kW)
    Fieldwork = Total_PV_price * r_Fieldwork
    Officework = Total_PV_price * r_Officework
    Other = Total_PV_price * r_Other
    Electrical_BoS = Total_PV_price * r_Electrical_BoS
    Structrual_BoS = Total_PV_price * r_Structrual_BoS
    Engineering_Costs = (Fieldwork + Officework + Other + Electrical_BoS + Structrual_BoS)

    # PV
    C_PV = Total_PV_price * r_PV  # Capital cost ($) per KW
    R_PV = Total_PV_price * r_PV  # Replacement Cost of PV modules Per KW

    # Inverter
    C_I = Total_PV_price * r_inverter  # Capital cost ($/kW)
    R_I = Total_PV_price * r_inverter  # Replacement cost ($/kW)

    return Engineering_Costs, C_PV, R_PV, C_I, R_I