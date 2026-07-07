#!/usr/bin/env python3
"""
SAMAPy OPTIMIZATION - COMPLETE WITH ALL ELECTRICITY & NATURAL GAS RATES
======================================================================
ALL parameters, ALL 8 electricity rate structures, ALL 6 NG rate structures
"""

import os
import sys
import shutil
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
from math import ceil

# ============================================================================
# NATURAL GAS CONVERSION FUNCTION
# ============================================================================
NG_energycontent = 10.97  # Energy content of 1m3 Natural Gas in kWh
Furnace_eff = 0.95  # Efficiency of heating furnace


def cm2kwh(value, multiplier_factor=1 / (NG_energycontent * Furnace_eff)):
    """Convert $/m3 to $/kWh for natural gas"""
    return value * multiplier_factor


class Config:
    """ALL configurable parameters - COMPLETE"""

    # ========================================================================
    # FILES
    # ========================================================================
    Eload_csv_path = "Eload.csv"
    METEO_csv_path = "METEO.csv"
    house_load_xlsx_path = "house_load.xlsx"

    # ========================================================================
    # PSO ALGORITHM
    # ========================================================================
    MaxIt = 200
    nPop = 50
    w = 1.0
    wdamp = 0.99
    c1 = 2.0
    c2 = 2.0
    Run_Time = 1

    # ========================================================================
    # PROJECT
    # ========================================================================
    n = 25
    year = 2023
    holidays = np.array([1, 51, 97, 100, 142, 182, 219, 248, 273, 282, 315, 359, 360])

    # ========================================================================
    # SYSTEM COMPONENTS
    # ========================================================================
    PV = 1
    WT = 0
    DG = 0
    Bat = 1
    Lead_acid = 0
    Li_ion = 1
    Grid = 1
    HP = 1
    HP_brand = 'Bosch'
    EV = 1
    NG_Grid = 0

    # ========================================================================
    # GRID SETTINGS
    # ========================================================================
    NEM = 1
    cap_option = 2
    cap_size = 0
    generation_cap = 150
    available_roof_surface = 40
    PVPanel_surface_per_rated_capacity = 5

    # ========================================================================
    # CONSTRAINTS
    # ========================================================================
    LPSP_max_rate = 0.000999
    RE_min_rate = 50
    EM = 0
    Budget = 200000

    # ========================================================================
    # ECONOMIC
    # ========================================================================
    n_ir_rate = 4.5
    e_ir_rate = 2.0
    Tax_rate = 0
    RE_incentives_rate = 30

    # ========================================================================
    # PV
    # ========================================================================
    fpv = 0.9
    Tcof = -0.3
    Tref = 25
    Tc_noct = 45
    Ta_noct = 20
    G_noct = 800
    gama = 0.9
    n_PV = 0.2182
    Gref = 1000
    L_PV = 25
    Ppv_r = 1.0
    azimuth = 180
    tilt = 33
    soiling = 5

    # PV Economics
    Pricing_method = 1
    Total_PV_price = 2682
    C_PV = 338
    R_PV = 338
    MO_PV = 30.36

    # Bottom-up
    Fieldwork = 178
    Officework = 696
    Other = 586
    Permiting_and_Inspection = 0
    Electrical_BoS = 333
    Structrual_BoS = 237
    Supply_Chain_costs = 0
    Profit_costs = 0
    Sales_tax = 0

    # ========================================================================
    # INVERTER
    # ========================================================================
    n_I = 0.96
    L_I = 25
    DC_AC_ratio = 1.99
    C_I = 314
    R_I = 314
    MO_I = 0

    # ========================================================================
    # BATTERY
    # ========================================================================
    SOC_min = 0.1
    SOC_max = 1.0
    SOC_initial = 0.5
    self_discharge_rate = 0
    L_B = 10
    C_B = 1450
    R_B = 1450
    MO_B = 10

    # Lead-Acid
    Cnom_Leadacid = 83.4
    Vnom_leadacid = 12
    ef_bat_leadacid = 0.8
    Q_lifetime_leadacid = 8000
    Ich_max_leadacid = 16.7
    alfa_battery_leadacid = 1.0
    c = 0.403
    k_lead_acid = 0.827

    # Li-ion
    Cnom_Li = 167
    Vnom_Li_ion = 6
    ef_bat_Li = 0.90
    Q_lifetime_Li = 3000
    Ich_max_Li_ion = 167
    Idch_max_Li_ion = 500
    alfa_battery_Li_ion = 1.0

    # ========================================================================
    # WIND TURBINE
    # ========================================================================
    h_hub = 17
    h0 = 43.6
    nw = 1.0
    v_cut_out = 25
    v_cut_in = 2.5
    v_rated = 9.5
    alfa_wind_turbine = 0.14
    L_WT = 20
    Pwt_r = 1.0
    C_WT = 1200
    R_WT = 1200
    MO_WT = 40

    # ========================================================================
    # DIESEL GENERATOR
    # ========================================================================
    LR_DG = 0.25
    a = 0.4388
    b = 0.1097
    TL_DG = 24000
    Cdg_r = 5.5
    C_DG = 818
    R_DG = 818
    MO_DG = 0.016
    C_fuel = 1.281
    C_fuel_adj_rate = 2
    CO2 = 2.29
    CO = 0
    NOx = 0
    SO2 = 0

    # ========================================================================
    # ELECTRIC VEHICLE
    # ========================================================================
    Tin = 17
    Tout = 8
    C_ev = 82
    SOCe_min = 0.03
    SOCe_max = 0.97
    SOCe_initial = 0.85
    Pev_max = 11
    Range_EV = 468
    Daily_trip = 68
    SOC_dep = 0.85
    n_e = 0.9
    self_discharge_rate_ev = 0
    L_EV_dis = 400000
    degradation_percent = 0.019
    step_km = 1000
    L_EV = 25
    treat_special_days_as_home = False
    Cost_EV = 0
    R_EVB = 9840
    MO_EV = 0

    # ========================================================================
    # HEAT PUMP
    # ========================================================================
    L_HP = 5
    Php_r = 1000
    C_HP = 109.5
    R_HP = 109.5
    MO_HP = 20

    # ========================================================================
    # CHARGER
    # ========================================================================
    L_CH = 25
    C_CH = 149.99
    R_CH = 149.99
    MO_CH = 0

    # ========================================================================
    # ELECTRICITY GRID RATES (1-8)
    # ========================================================================
    rateStructure = 7

    # Fixed expenses
    Annual_expenses = 0
    Grid_sale_tax_rate = 0.986
    Grid_Tax_amount = 0.0016
    Grid_credit = 58.23 * 2
    NEM_fee = 0

    # Grid escalation
    Grid_escalation_projection = 1  # 1=Flat, 2=Yearly array
    Grid_escalation_rate_flat = 2.0
    Grid_escalation_rate_yearly = np.array([5.7, 5.7, 5.7, 5.7, 5.7, 5.7, 5.7, 5.7, 5.7, 5.7,
                                            5.7, 5.7, 5.7, 5.7, 5.7, 5.7, 5.7, 5.7, 5.7, 5.7,
                                            5.7, 5.7, 5.7, 5.7, 5.7])

    # Monthly Service Charge
    Monthly_fixed_charge_system = 1
    SC_flat = 15.0
    SC_1 = 34.29
    Limit_SC_1 = 800
    SC_2 = 46.54
    Limit_SC_2 = 1500
    SC_3 = 66.29
    Limit_SC_3 = 1500
    SC_4 = 66.29

    # 1: Flat
    flatPrice = 0.2

    # 2: Seasonal
    seasonalPrices = np.array([0.0719, 0.0540])  # [summer, winter]
    season = np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0])  # 1=Summer

    # 3: Monthly
    monthlyPrices = np.array([0.54207, 0.53713, 0.38689, 0.30496, 0.28689,
                              0.28168, 0.30205, 0.28956, 0.26501, 0.26492,
                              0.3108, 0.40715])

    # 4: Tiered
    tieredPrices = np.array([0.1018, 0.1175, 0.1175])
    tierMax = np.array([300, 999999, 999999])

    # 5: Seasonal Tiered
    seasonalTieredPrices = np.array([[0.075, 0.091, 0.091],
                                     [0.075, 0.091, 0.091]])
    seasonalTierMax = np.array([[600, 999999, 999999],
                                [1000, 999999, 999999]])
    season_tiered = np.array([0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0])

    # 6: Monthly Tiered
    monthlyTieredPrices = np.array([
        [0.404, 0.509, 0.509],
        [0.404, 0.509, 0.509],
        [0.404, 0.509, 0.509],
        [0.404, 0.509, 0.509],
        [0.404, 0.509, 0.509],
        [0.404, 0.509, 0.509],
        [0.404, 0.509, 0.509],
        [0.404, 0.509, 0.509],
        [0.404, 0.509, 0.509],
        [0.404, 0.509, 0.509],
        [0.404, 0.509, 0.509],
        [0.404, 0.509, 0.509]
    ])
    monthlyTierLimits = np.array([
        [343, 999999, 999999],
        [343, 999999, 999999],
        [343, 999999, 999999],
        [343, 999999, 999999],
        [343, 999999, 999999],
        [234, 999999, 999999],
        [234, 999999, 999999],
        [234, 999999, 999999],
        [234, 999999, 999999],
        [234, 999999, 999999],
        [343, 999999, 999999],
        [343, 999999, 999999]
    ])

    # 7: TOU
    onPrice = np.array([0.61, 0.38])
    midPrice = np.array([0.45, 0.36])
    offPrice = np.array([0.4, 0.35])
    onHours = np.array([[16, 17, 18, 19, 20], [16, 17, 18, 19, 20]], dtype=object)
    midHours = np.array([[15, 21, 22, 23], [15, 21, 22, 23]], dtype=object)
    treat_special_days_as_offpeak = False

    # 8: Ultra-Low TOU
    onPrice_UL = np.array([0.284, 0.284])
    midPrice_UL = np.array([0.122, 0.122])
    offPrice_UL = np.array([0.076, 0.076])
    ultraLowPrice = np.array([0.028, 0.028])
    onHours_UL = np.array([[16, 17, 18, 19, 20], [16, 17, 18, 19, 20]], dtype=object)
    midHours_UL = np.array([[7, 8, 9, 10, 11, 12, 13, 14, 15, 21, 22],
                            [7, 8, 9, 10, 11, 12, 13, 14, 15, 21, 22]], dtype=object)
    ultraLowHours = np.array([[23, 0, 1, 2, 3, 4, 5, 6],
                              [23, 0, 1, 2, 3, 4, 5, 6]], dtype=object)
    season_UL = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
    treat_special_days_as_offpeak_UL = True

    # ========================================================================
    # GRID SELL (EXPORT)
    # ========================================================================
    sellStructure = 2
    C_sell_flat = 0.1
    monthlysellprices = np.array([0.05799, 0.04829, 0.04621, 0.04256, 0.04030,
                                  0.03991, 0.03963, 0.03976, 0.03781, 0.03656,
                                  0.03615, 0.03461])

    # Grid buy/sell limits
    Pbuy_max = 50
    Psell_max = 50

    # Grid emissions
    E_CO2 = 0.0
    E_SO2 = 0.0
    E_NOx = 0.0

    # ========================================================================
    # NATURAL GAS RATES (1-6)
    # ========================================================================
    rateStructure_NG = 1

    # Fixed expenses
    Annual_expenses_NG = 0
    Grid_sale_tax_rate_NG = 13
    Grid_Tax_amount_NG = cm2kwh(0.11)
    Grid_credit_NG = cm2kwh(0)

    # NG Grid escalation
    Grid_escalation_projection_NG = 1
    Grid_escalation_rate_flat_NG = 2.0
    Grid_escalation_rate_yearly_NG = np.array([5.7, 5.7, 5.7, 5.7, 5.7, 5.7, 5.7, 5.7, 5.7, 5.7,
                                               5.7, 5.7, 5.7, 5.7, 5.7, 5.7, 5.7, 5.7, 5.7, 5.7,
                                               5.7, 5.7, 5.7, 5.7, 5.7])

    # NG Monthly Service Charge
    Monthly_fixed_charge_system_NG = 1
    SC_flat_NG = 18.59

    # 1: Flat
    flatPrice_NG = cm2kwh(0.28)

    # 2: Seasonal
    seasonalPrices_NG = cm2kwh(np.array([0.0719, 0.0540]))
    season_NG = np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0])

    # 3: Monthly
    monthlyPrices_NG = cm2kwh(np.array([0.54207, 0.53713, 0.38689, 0.30496, 0.28689,
                                        0.28168, 0.30205, 0.28956, 0.26501, 0.26492,
                                        0.3108, 0.40715]))

    # 4: Tiered
    tieredPrices_NG = cm2kwh(np.array([0.1018, 0.1175, 0.1175]))
    tierMax_NG = np.array([300, 999999, 999999])

    # 5: Seasonal Tiered
    seasonalTieredPrices_NG = cm2kwh(np.array([[0.075, 0.091, 0.091],
                                               [0.075, 0.091, 0.091]]))
    seasonalTierMax_NG = np.array([[600, 999999, 999999],
                                   [1000, 999999, 999999]])
    season_tiered_NG = np.array([0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0])

    # 6: Monthly Tiered
    monthlyTieredPrices_NG = cm2kwh(np.array([
        [0.404, 0.509, 0.509],
        [0.404, 0.509, 0.509],
        [0.404, 0.509, 0.509],
        [0.404, 0.509, 0.509],
        [0.404, 0.509, 0.509],
        [0.404, 0.509, 0.509],
        [0.404, 0.509, 0.509],
        [0.404, 0.509, 0.509],
        [0.404, 0.509, 0.509],
        [0.404, 0.509, 0.509],
        [0.404, 0.509, 0.509],
        [0.404, 0.509, 0.509]
    ]))
    monthlyTierLimits_NG = np.array([
        [343, 999999, 999999],
        [343, 999999, 999999],
        [343, 999999, 999999],
        [343, 999999, 999999],
        [343, 999999, 999999],
        [234, 999999, 999999],
        [234, 999999, 999999],
        [234, 999999, 999999],
        [234, 999999, 999999],
        [234, 999999, 999999],
        [343, 999999, 999999],
        [343, 999999, 999999]
    ])


def copy_files_to_content(config):
    """Copy data files to samapy/samapy/content/"""
    print("\n" + "=" * 80)
    print("STEP 1: COPYING DATA FILES")
    print("=" * 80)

    from samapy import get_content_path

    eload_dest = get_content_path('Eload.csv')
    eload_src = Path(config.Eload_csv_path)

    if not eload_src.exists():
        print(f"ERROR: {eload_src} not found!")
        sys.exit(1)

    shutil.copy2(eload_src, eload_dest)
    df = pd.read_csv(eload_dest, header=None)
    csv_total = df[0].sum()
    print(f"\n✓ Eload.csv → {eload_dest}")
    print(f"  CSV Total: {csv_total:.0f} kWh/year")

    meteo_dest = get_content_path('METEO.csv')
    meteo_src = Path(config.METEO_csv_path)
    if meteo_src.exists():
        shutil.copy2(meteo_src, meteo_dest)
        print(f"✓ METEO.csv → {meteo_dest}")

    if config.HP == 1:
        house_dest = get_content_path('house_load.xlsx')
        house_src = Path(config.house_load_xlsx_path)
        if house_src.exists():
            shutil.copy2(house_src, house_dest)
            print(f"✓ house_load.xlsx → {house_dest}")

    print("=" * 80)
    return csv_total


def apply_all_parameters(config, csv_total):
    """Apply ALL parameters - HP model runs ONLY ONCE"""
    print("\n" + "=" * 80)
    print("STEP 2: APPLYING ALL PARAMETERS")
    print("=" * 80)

    # Clear modules
    modules_to_clear = [k for k in list(sys.modules.keys()) if k.startswith('samapy.')]
    for mod in modules_to_clear:
        del sys.modules[mod]

    # Set environment variable to skip HP in Input_Data.py
    os.environ['SAMAPy_SKIP_HP_CALC'] = '1'

    from samapy.core.Input_Data import InData

    # Clean up
    if 'SAMAPy_SKIP_HP_CALC' in os.environ:
        del os.environ['SAMAPy_SKIP_HP_CALC']

    print("\n✓ InData imported")

    # ========================================================================
    # Components
    # ========================================================================
    InData.PV = config.PV
    InData.WT = config.WT
    InData.DG = config.DG
    InData.Bat = config.Bat
    InData.Lead_acid = config.Lead_acid
    InData.Li_ion = config.Li_ion
    InData.Grid = config.Grid
    InData.HP = config.HP
    InData.HP_brand = config.HP_brand
    InData.EV = config.EV
    InData.NG_Grid = config.NG_Grid

    # PSO
    InData.MaxIt = config.MaxIt
    InData.nPop = config.nPop
    InData.w = config.w
    InData.wdamp = config.wdamp
    InData.c1 = config.c1
    InData.c2 = config.c2
    InData.Run_Time = config.Run_Time

    # Project
    InData.n = config.n
    InData.year = config.year
    InData.holidays = config.holidays

    # Grid settings
    InData.NEM = config.NEM
    InData.cap_option = config.cap_option
    InData.cap_size = config.cap_size
    InData.generation_cap = config.generation_cap

    # Constraints
    InData.LPSP_max_rate = config.LPSP_max_rate
    InData.LPSP_max = config.LPSP_max_rate / 100
    InData.RE_min_rate = config.RE_min_rate
    InData.RE_min = config.RE_min_rate / 100
    InData.EM = config.EM
    InData.Budget = config.Budget

    # Economic
    InData.n_ir_rate = config.n_ir_rate
    InData.n_ir = config.n_ir_rate / 100
    InData.e_ir_rate = config.e_ir_rate
    InData.e_ir = config.e_ir_rate / 100
    InData.ir = (InData.n_ir - InData.e_ir) / (1 + InData.e_ir)
    InData.Tax_rate = config.Tax_rate
    InData.System_Tax = config.Tax_rate / 100
    InData.RE_incentives_rate = config.RE_incentives_rate
    InData.RE_incentives = config.RE_incentives_rate / 100

    # PV
    InData.fpv = config.fpv
    InData.Tcof = config.Tcof
    InData.Tref = config.Tref
    InData.Tc_noct = config.Tc_noct
    InData.Ta_noct = config.Ta_noct
    InData.G_noct = config.G_noct
    InData.gama = config.gama
    InData.n_PV = config.n_PV
    InData.Gref = config.Gref
    InData.L_PV = config.L_PV
    InData.Ppv_r = config.Ppv_r
    InData.RT_PV = ceil(config.n / config.L_PV) - 1

    InData.Pricing_method = config.Pricing_method
    if config.Pricing_method == 1:
        from samapy.utilities.top_down import top_down_pricing
        InData.Engineering_Costs, InData.C_PV, InData.R_PV, InData.C_I, InData.R_I = \
            top_down_pricing(config.Total_PV_price)
    else:
        InData.Engineering_Costs = (config.Sales_tax + config.Profit_costs +
                                    config.Fieldwork + config.Officework + config.Other +
                                    config.Permiting_and_Inspection + config.Electrical_BoS +
                                    config.Structrual_BoS + config.Supply_Chain_costs)
        InData.C_PV = config.C_PV
        InData.R_PV = config.R_PV
        InData.C_I = config.C_I
        InData.R_I = config.R_I
    InData.MO_PV = config.MO_PV

    # Inverter
    InData.n_I = config.n_I
    InData.L_I = config.L_I
    InData.DC_AC_ratio = config.DC_AC_ratio
    InData.MO_I = config.MO_I
    InData.RT_I = ceil(config.n / config.L_I) - 1

    # Battery
    InData.SOC_min = config.SOC_min
    InData.SOC_max = config.SOC_max
    InData.SOC_initial = config.SOC_initial
    InData.self_discharge_rate = config.self_discharge_rate
    InData.L_B = config.L_B
    InData.C_B = config.C_B
    InData.R_B = config.R_B
    InData.MO_B = config.MO_B
    InData.RT_B = ceil(config.n / config.L_B) - 1

    if config.Lead_acid == 1:
        InData.Cnom_Leadacid = config.Cnom_Leadacid
        InData.Vnom_leadacid = config.Vnom_leadacid
        InData.ef_bat_leadacid = config.ef_bat_leadacid
        InData.Q_lifetime_leadacid = config.Q_lifetime_leadacid
        InData.Ich_max_leadacid = config.Ich_max_leadacid
        InData.alfa_battery_leadacid = config.alfa_battery_leadacid
        InData.c = config.c
        InData.k_lead_acid = config.k_lead_acid
        InData.Cbt_r = (config.Vnom_leadacid * config.Cnom_Leadacid) / 1000

    if config.Li_ion == 1:
        InData.Cnom_Li = config.Cnom_Li
        InData.Vnom_Li_ion = config.Vnom_Li_ion
        InData.ef_bat_Li = config.ef_bat_Li
        InData.Q_lifetime_Li = config.Q_lifetime_Li
        InData.Ich_max_Li_ion = config.Ich_max_Li_ion
        InData.Idch_max_Li_ion = config.Idch_max_Li_ion
        InData.alfa_battery_Li_ion = config.alfa_battery_Li_ion
        InData.Cbt_r = (config.Vnom_Li_ion * config.Cnom_Li) / 1000

    # Wind Turbine
    InData.h_hub = config.h_hub
    InData.h0 = config.h0
    InData.nw = config.nw
    InData.v_cut_out = config.v_cut_out
    InData.v_cut_in = config.v_cut_in
    InData.v_rated = config.v_rated
    InData.alfa_wind_turbine = config.alfa_wind_turbine
    InData.L_WT = config.L_WT
    InData.Pwt_r = config.Pwt_r
    InData.C_WT = config.C_WT
    InData.R_WT = config.R_WT
    InData.MO_WT = config.MO_WT
    InData.RT_WT = ceil(config.n / config.L_WT) - 1

    # Diesel Generator
    InData.LR_DG = config.LR_DG
    InData.a = config.a
    InData.b = config.b
    InData.TL_DG = config.TL_DG
    InData.Cdg_r = config.Cdg_r
    InData.C_DG = config.C_DG
    InData.R_DG = config.R_DG
    InData.MO_DG = config.MO_DG
    InData.C_fuel = config.C_fuel
    InData.C_fuel_adj_rate = config.C_fuel_adj_rate
    InData.C_fuel_adj = config.C_fuel_adj_rate / 100
    InData.CO2 = config.CO2
    InData.CO = config.CO
    InData.NOx = config.NOx
    InData.SO2 = config.SO2

    # EV (always set)
    InData.Tin = config.Tin
    InData.Tout = config.Tout
    InData.C_ev = config.C_ev
    InData.SOCe_min = config.SOCe_min
    InData.SOCe_max = config.SOCe_max
    InData.C_ev_usable = config.C_ev * (config.SOCe_max - config.SOCe_min)
    InData.SOCe_initial = config.SOCe_initial
    InData.Pev_max = config.Pev_max
    InData.Range_EV = config.Range_EV
    InData.Daily_trip = config.Daily_trip
    InData.SOC_dep = config.SOC_dep
    InData.SOC_arr = config.SOC_dep - ((config.Daily_trip * InData.C_ev_usable) /
                                       (config.Range_EV * config.C_ev))
    InData.n_e = config.n_e
    InData.self_discharge_rate_ev = config.self_discharge_rate_ev
    InData.L_EV_dis = config.L_EV_dis
    InData.degradation_percent = config.degradation_percent
    InData.step_km = config.step_km
    InData.L_EV = config.L_EV
    InData.treat_special_days_as_home = config.treat_special_days_as_home
    InData.Cost_EV = config.Cost_EV
    InData.R_EVB = config.R_EVB
    InData.MO_EV = config.MO_EV

    from samapy.utilities.Ev_Battery_Throughput import calculate_ev_battery_throughput
    _, InData.Q_lifetime_ev = calculate_ev_battery_throughput(
        InData.C_ev_usable, config.degradation_percent, config.L_EV_dis,
        config.Range_EV, config.step_km
    )
    InData.RT_EV = ceil(config.n / config.L_EV) - 1

    if config.EV == 1:
        from samapy.utilities.EV_Presence import determine_EV_presence
        InData.EVp = determine_EV_presence(
            config.year, config.Tout, config.Tin,
            config.holidays, config.treat_special_days_as_home
        )

    # Heat Pump (always set)
    InData.L_HP = config.L_HP
    InData.Php_r = config.Php_r
    InData.C_HP = config.C_HP
    InData.R_HP = config.R_HP
    InData.MO_HP = config.MO_HP
    InData.RT_HP = ceil(config.n / config.L_HP) - 1

    # Charger
    InData.L_CH = config.L_CH
    InData.C_CH = config.C_CH
    InData.R_CH = config.R_CH
    InData.MO_CH = config.MO_CH
    InData.RT_CH = ceil(config.n / config.L_CH) - 1

    # ========================================================================
    # ELECTRICITY GRID RATES (1-8)
    # ========================================================================
    print("\n" + "-" * 80)
    print(f"Applying ELECTRICITY rates (Structure {config.rateStructure})...")
    print("-" * 80)

    # Fixed expenses
    InData.Annual_expenses = config.Annual_expenses
    InData.Grid_sale_tax_rate = config.Grid_sale_tax_rate
    InData.Grid_Tax = config.Grid_sale_tax_rate / 100
    InData.Grid_Tax_amount = config.Grid_Tax_amount
    InData.Grid_credit = config.Grid_credit
    InData.NEM_fee = config.NEM_fee

    # Grid escalation
    if config.Grid_escalation_projection == 1:
        InData.Grid_escalation_rate = np.full(25, config.Grid_escalation_rate_flat)
    else:
        InData.Grid_escalation_rate = config.Grid_escalation_rate_yearly
    InData.Grid_escalation = InData.Grid_escalation_rate / 100

    # Service charge
    InData.Monthly_fixed_charge_system = config.Monthly_fixed_charge_system
    if config.Monthly_fixed_charge_system == 1:
        InData.SC_flat = config.SC_flat
        InData.Service_charge = np.ones(12) * config.SC_flat
    else:
        InData.SC_1 = config.SC_1
        InData.Limit_SC_1 = config.Limit_SC_1
        InData.SC_2 = config.SC_2
        InData.Limit_SC_2 = config.Limit_SC_2
        InData.SC_3 = config.SC_3
        InData.Limit_SC_3 = config.Limit_SC_3
        InData.SC_4 = config.SC_4
        from samapy.pricing.service_charge import service_charge
        InData.Service_charge = service_charge(
            InData.daysInMonth, InData.Eload_Previous,
            config.Limit_SC_1, config.SC_1,
            config.Limit_SC_2, config.SC_2,
            config.Limit_SC_3, config.SC_3, config.SC_4
        )

    # Hourly rates
    if config.rateStructure == 1:  # Flat
        InData.flatPrice = config.flatPrice
        from samapy.pricing.calcFlatRate import calcFlatRate
        InData.Cbuy = calcFlatRate(config.flatPrice)
        print(f"  Flat: ${config.flatPrice}/kWh")

    elif config.rateStructure == 2:  # Seasonal
        InData.seasonalPrices = config.seasonalPrices
        InData.season = config.season
        from samapy.pricing.calcSeasonalRate import calcSeasonalRate
        InData.Cbuy = calcSeasonalRate(config.seasonalPrices, config.season, InData.daysInMonth)
        print(f"  Seasonal: Summer=${config.seasonalPrices[0]}, Winter=${config.seasonalPrices[1]}")

    elif config.rateStructure == 3:  # Monthly
        InData.monthlyPrices = config.monthlyPrices
        from samapy.pricing.calcMonthlyRate import calcMonthlyRate
        InData.Cbuy = calcMonthlyRate(config.monthlyPrices, InData.daysInMonth)
        print(f"  Monthly: 12 different rates")

    elif config.rateStructure == 4:  # Tiered
        InData.tieredPrices = config.tieredPrices
        InData.tierMax = config.tierMax
        from samapy.pricing.calcTieredRate import calcTieredRate
        InData.Cbuy = calcTieredRate(config.tieredPrices, config.tierMax, InData.Eload, InData.daysInMonth)
        print(f"  Tiered: {len(config.tieredPrices)} tiers")

    elif config.rateStructure == 5:  # Seasonal Tiered
        InData.seasonalTieredPrices = config.seasonalTieredPrices
        InData.seasonalTierMax = config.seasonalTierMax
        InData.season = config.season_tiered
        from samapy.pricing.calcSeasonalTieredRate import calcSeasonalTieredRate
        InData.Cbuy = calcSeasonalTieredRate(
            config.seasonalTieredPrices, config.seasonalTierMax, InData.Eload, config.season_tiered
        )
        print(f"  Seasonal Tiered: 2 seasons x {config.seasonalTieredPrices.shape[1]} tiers")

    elif config.rateStructure == 6:  # Monthly Tiered
        InData.monthlyTieredPrices = config.monthlyTieredPrices
        InData.monthlyTierLimits = config.monthlyTierLimits
        from samapy.pricing.calcMonthlyTieredRate import calcMonthlyTieredRate
        InData.Cbuy = calcMonthlyTieredRate(config.monthlyTieredPrices, config.monthlyTierLimits, InData.Eload)
        print(f"  Monthly Tiered: 12 months x {config.monthlyTieredPrices.shape[1]} tiers")

    elif config.rateStructure == 7:  # TOU
        InData.onPrice = config.onPrice
        InData.midPrice = config.midPrice
        InData.offPrice = config.offPrice
        InData.onHours = config.onHours
        InData.midHours = config.midHours
        InData.season = config.season
        InData.treat_special_days_as_offpeak = config.treat_special_days_as_offpeak
        from samapy.pricing.calcTouRate import calcTouRate
        InData.Cbuy = calcTouRate(
            config.year, config.onPrice, config.midPrice, config.offPrice,
            config.onHours, config.midHours, config.season, InData.daysInMonth,
            config.holidays, config.treat_special_days_as_offpeak
        )
        print(f"  TOU: On=${config.onPrice[0]}, Mid=${config.midPrice[0]}, Off=${config.offPrice[0]}")

    elif config.rateStructure == 8:  # Ultra-Low TOU
        InData.onPrice = config.onPrice_UL
        InData.midPrice = config.midPrice_UL
        InData.offPrice = config.offPrice_UL
        InData.ultraLowPrice = config.ultraLowPrice
        InData.onHours = config.onHours_UL
        InData.midHours = config.midHours_UL
        InData.ultraLowHours = config.ultraLowHours
        InData.season = config.season_UL
        InData.treat_special_days_as_offpeak = config.treat_special_days_as_offpeak_UL
        from samapy.pricing.calcULTouRate import calcULTouRate
        InData.Cbuy = calcULTouRate(
            config.year, config.onPrice_UL, config.midPrice_UL, config.offPrice_UL, config.ultraLowPrice,
            config.onHours_UL, config.midHours_UL, config.ultraLowHours, config.season_UL,
            InData.daysInMonth, config.holidays, config.treat_special_days_as_offpeak_UL
        )
        print(
            f"  Ultra-Low TOU: On=${config.onPrice_UL[0]}, Mid=${config.midPrice_UL[0]}, Off=${config.offPrice_UL[0]}, UL=${config.ultraLowPrice[0]}")

    # Grid sell
    InData.sellStructure = config.sellStructure
    if config.sellStructure == 1:
        InData.Csell = np.full(8760, config.C_sell_flat)
    elif config.sellStructure == 2:
        InData.monthlysellprices = config.monthlysellprices
        from samapy.pricing.calcMonthlyRate import calcMonthlyRate
        InData.Csell = calcMonthlyRate(config.monthlysellprices, InData.daysInMonth)
    elif config.sellStructure == 3:
        InData.Csell = InData.Cbuy

    # Grid constraints
    InData.Pbuy_max = config.Pbuy_max
    InData.Psell_max = config.Psell_max

    # Grid emissions
    InData.E_CO2 = config.E_CO2
    InData.E_SO2 = config.E_SO2
    InData.E_NOx = config.E_NOx

    # ========================================================================
    # NATURAL GAS RATES (1-6)
    # ========================================================================
    print("\n" + "-" * 80)
    print(f"Applying NATURAL GAS rates (Structure {config.rateStructure_NG})...")
    print("-" * 80)

    InData.rateStructure_NG = config.rateStructure_NG
    InData.Annual_expenses_NG = config.Annual_expenses_NG
    InData.Grid_sale_tax_rate_NG = config.Grid_sale_tax_rate_NG
    InData.Grid_Tax_NG = config.Grid_sale_tax_rate_NG / 100
    InData.Grid_Tax_amount_NG = config.Grid_Tax_amount_NG
    InData.Grid_credit_NG = config.Grid_credit_NG

    # NG escalation
    if config.Grid_escalation_projection_NG == 1:
        InData.Grid_escalation_rate_NG = np.full(25, config.Grid_escalation_rate_flat_NG)
    else:
        InData.Grid_escalation_rate_NG = config.Grid_escalation_rate_yearly_NG
    InData.Grid_escalation_NG = InData.Grid_escalation_rate_NG / 100

    # NG Service charge
    InData.Monthly_fixed_charge_system_NG = config.Monthly_fixed_charge_system_NG
    if config.Monthly_fixed_charge_system_NG == 1:
        InData.SC_flat_NG = config.SC_flat_NG
        InData.Service_charge_NG = np.ones(12) * config.SC_flat_NG

    # NG hourly rates
    if config.rateStructure_NG == 1:  # Flat
        InData.flatPrice_NG = config.flatPrice_NG
        from samapy.pricing.calcFlatRate import calcFlatRate
        InData.Cbuy_NG = calcFlatRate(config.flatPrice_NG)
        print(f"  NG Flat: ${config.flatPrice_NG:.4f}/kWh")

    elif config.rateStructure_NG == 2:  # Seasonal
        InData.seasonalPrices_NG = config.seasonalPrices_NG
        InData.season_NG = config.season_NG
        from samapy.pricing.calcSeasonalRate import calcSeasonalRate
        InData.Cbuy_NG = calcSeasonalRate(config.seasonalPrices_NG, config.season_NG, InData.daysInMonth)
        print(f"  NG Seasonal")

    elif config.rateStructure_NG == 3:  # Monthly
        InData.monthlyPrices_NG = config.monthlyPrices_NG
        from samapy.pricing.calcMonthlyRate import calcMonthlyRate
        InData.Cbuy_NG = calcMonthlyRate(config.monthlyPrices_NG, InData.daysInMonth)
        print(f"  NG Monthly")

    elif config.rateStructure_NG == 4:  # Tiered
        InData.tieredPrices_NG = config.tieredPrices_NG
        InData.tierMax_NG = config.tierMax_NG
        from samapy.pricing.calcTieredRate import calcTieredRate
        InData.Cbuy_NG = calcTieredRate(config.tieredPrices_NG, config.tierMax_NG, InData.Tload, InData.daysInMonth)
        print(f"  NG Tiered")

    elif config.rateStructure_NG == 5:  # Seasonal Tiered
        InData.seasonalTieredPrices_NG = config.seasonalTieredPrices_NG
        InData.seasonalTierMax_NG = config.seasonalTierMax_NG
        InData.season_NG = config.season_tiered_NG
        from samapy.pricing.calcSeasonalTieredRate import calcSeasonalTieredRate
        InData.Cbuy_NG = calcSeasonalTieredRate(
            config.seasonalTieredPrices_NG, config.seasonalTierMax_NG, InData.Tload, config.season_tiered_NG
        )
        print(f"  NG Seasonal Tiered")

    elif config.rateStructure_NG == 6:  # Monthly Tiered
        InData.monthlyTieredPrices_NG = config.monthlyTieredPrices_NG
        InData.monthlyTierLimits_NG = config.monthlyTierLimits_NG
        from samapy.pricing.calcMonthlyTieredRate import calcMonthlyTieredRate
        InData.Cbuy_NG = calcMonthlyTieredRate(config.monthlyTieredPrices_NG, config.monthlyTierLimits_NG, InData.Tload)
        print(f"  NG Monthly Tiered")

    print("\n" + "=" * 80)
    print("✅ ALL PARAMETERS APPLIED!")
    print("=" * 80)
    print(f"  PSO: MaxIt={InData.MaxIt}, nPop={InData.nPop}")
    print(f"  Components: HP={InData.HP}, EV={InData.EV}")
    print(f"  Load: {np.sum(InData.Eload):.0f} kWh/year")
    print(f"  Electricity rate: {config.rateStructure}")
    print(f"  NG rate: {config.rateStructure_NG}")
    print("=" * 80)

    return InData


def run_optimization(config):
    """Main optimization"""
    print("\n" + "=" * 80)
    print("SAMAPy OPTIMIZATION - COMPLETE VERSION")
    print("=" * 80)

    csv_total = copy_files_to_content(config)
    InData = apply_all_parameters(config, csv_total)

    # Verification
    print("\n" + "=" * 80)
    print("FINAL VERIFICATION")
    print("=" * 80)
    print(f"Config.HP = {config.HP}, InData.HP = {InData.HP}")
    print(f"CSV = {csv_total:.0f}, InData.Eload = {np.sum(InData.Eload):.0f} kWh/year")

    if config.HP == 0:
        diff = abs(np.sum(InData.Eload) - csv_total)
        if diff < 100:
            print(f"✅ CORRECT! HP=0 and load matches (diff={diff:.0f})")
        else:
            print(f"❌ ERROR! HP=0 but load doesn't match (diff={diff:.0f})")
            return

    print("=" * 80)

    # Run optimization
    print("\n" + "=" * 80)
    print("RUNNING OPTIMIZATION")
    print("=" * 80)

    proceed = input(f"\nRun {config.MaxIt} iterations? (y/n): ")
    if proceed.lower() != 'y':
        print("Cancelled.")
        return

    if config.HP == 1:
        print(f"\n🔧 Calculating HP load with {config.HP_brand}...")

        if config.HP_brand == 'Goodman':
            from samapy.models.BB_HP_Goodman import Heat_Pump_Model
            InData.Eload_hp, _, _, _, _, _, _ = Heat_Pump_Model(
                InData.T, InData.P / 10, InData.Hload, InData.Cload
            )
        elif config.HP_brand == 'Bosch':
            from samapy.models.BB_HP_Bosch import Heat_Pump_Model
            InData.Eload_hp, _, _, _, _, _, _ = Heat_Pump_Model(
                InData.T, InData.P / 10, InData.Hload, InData.Cload
            )

        # Update total load
        if isinstance(InData.Eload_hp, pd.Series):
            InData.Eload = InData.Eload_eh + InData.Eload_hp.to_numpy()
            InData.Eload_Previous = InData.Eload_eh + InData.Eload_hp.to_numpy()
        else:
            InData.Eload = np.array(InData.Eload_eh) + np.array(InData.Eload_hp)
            InData.Eload_Previous = np.array(InData.Eload_eh) + np.array(InData.Eload_hp)

        print(f"   ✓ HP load: {np.sum(InData.Eload_hp):.0f} kWh/year")

    start = datetime.now()
    from samapy.optimizers.swarm import Swarm
    optimizer = Swarm()
    optimizer.optimize()
    duration = (datetime.now() - start).total_seconds() / 60

    best_idx = np.argmin(optimizer.solution_best_costs)
    solution = optimizer.solution_best_positions[best_idx]
    npc = optimizer.solution_best_costs[best_idx]

    print(f"\n✓ Completed in {duration:.1f} minutes")
    print(f"\nOptimal System:")
    print(f"  PV: {solution[0]:.0f} panels")
    print(f"  Battery: {solution[2]:.0f} units")
    print(f"  NPC: ${npc:,.0f}")
    print("=" * 80)


if __name__ == "__main__":
    config = Config()
    run_optimization(config)
