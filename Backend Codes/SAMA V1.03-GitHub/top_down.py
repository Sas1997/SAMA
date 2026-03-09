def top_down_pricing(Total_PV_price):

    # Pricing method 1/top down

    # NREL percentages
    r_PV = 0.1812
    r_inverter = 0.1492
    r_Installation_cost = 0.0542
    r_Overhead = 0.0881
    r_Sales_and_marketing = 0.1356
    r_Permiting_and_Inspection = 0.0712
    r_Electrical_BoS = 0.1254
    r_Structrual_BoS = 0.0542
    r_Profit_costs = 0.1152
    r_Sales_tax = 0.0271
    r_Supply_Chain_costs = 0

    # Engineering Costs (Per/kW)
    Installation_cost = Total_PV_price * r_Installation_cost
    Overhead = Total_PV_price * r_Overhead
    Sales_and_marketing = Total_PV_price * r_Sales_and_marketing
    Permiting_and_Inspection = Total_PV_price * r_Permiting_and_Inspection
    Electrical_BoS = Total_PV_price * r_Electrical_BoS
    Structrual_BoS = Total_PV_price * r_Structrual_BoS
    Profit_costs = Total_PV_price * r_Profit_costs
    Sales_tax = Total_PV_price * r_Sales_tax
    Supply_Chain_costs = Total_PV_price * r_Supply_Chain_costs
    Engineering_Costs = (Sales_tax + Profit_costs + Installation_cost + Overhead + Sales_and_marketing + Permiting_and_Inspection + Electrical_BoS + Structrual_BoS + Supply_Chain_costs)

    # PV
    C_PV = Total_PV_price * r_PV  # Capital cost ($) per KW
    R_PV = Total_PV_price * r_PV  # Replacement Cost of PV modules Per KW

    # Inverter
    C_I = Total_PV_price * r_inverter  # Capital cost ($/kW)
    R_I = Total_PV_price * r_inverter  # Replacement cost ($/kW)

    return Engineering_Costs, C_PV, R_PV, C_I, R_I, r_Sales_tax