#!/usr/bin/env python3

"""
SAMAPy MASTER CONFIGURATION WIZARD

This wizard creates a complete configuration file that aligns perfectly with Input_Data.py structure.
All parameter names, structures, and data types match the backend requirements.

"""

import sys
import os
import questionary
import yaml
import numpy as np
from pathlib import Path
import subprocess
import shutil


def copy_to_content(source_file, target_filename):
    """Copy user file to samapy/content/ folder"""
    content_dir = Path.cwd() / 'samapy' / 'content'
    content_dir.mkdir(parents=True, exist_ok=True)

    source = Path(source_file)
    target = content_dir / target_filename

    if source.exists():
        shutil.copy2(source, target)
        print(f"✅ Copied {source_file} → {target}")
        return True
    else:
        print(f"❌ File not found: {source_file}")
        return False


def setup_directories():
    """Ask only where results should be saved. Config saves in cwd."""
    print("\nOutput Directory Setup\n")
    dirs = {'output_dir': questionary.text("Where should results be saved?", default=str(Path.cwd() / 'samapy_outputs')).ask()}
    input_dir = Path.cwd()  # config YAML always saved in cwd
    output_dir = Path(dirs['output_dir'])
    output_dir.mkdir(parents=True, exist_ok=True)
    return input_dir, output_dir


# def top_down_pricing(Total_PV_price):
#     """Calculate costs using top-down NREL percentages"""
#     # NREL percentages
#     r_PV = 0.126
#     r_inverter = 0.1171
#     r_Fieldwork = 0.06537
#     r_Officework = 0.2595
#     r_Other = 0.2185
#     r_Electrical_BoS = 0.1242
#     r_Structrual_BoS = 0.08837
#
#     # Engineering Costs (Per/kW)
#     Fieldwork = Total_PV_price * r_Fieldwork
#     Officework = Total_PV_price * r_Officework
#     Other = Total_PV_price * r_Other
#     Electrical_BoS = Total_PV_price * r_Electrical_BoS
#     Structrual_BoS = Total_PV_price * r_Structrual_BoS
#     Engineering_Costs = (Fieldwork + Officework + Other + Electrical_BoS + Structrual_BoS)
#
#     # PV
#     C_PV = Total_PV_price * r_PV
#     R_PV = Total_PV_price * r_PV
#
#     # Inverter
#     C_I = Total_PV_price * r_inverter
#     R_I = Total_PV_price * r_inverter
#
#     return Engineering_Costs, C_PV, R_PV, C_I, R_I


def run_sub_wizard(script_name):
    """Run a sub-wizard and load its output"""
    print(f"\n→ Launching {script_name}...")

    script_dir = Path(__file__).parent
    sub_wizard_path = script_dir / script_name

    if not sub_wizard_path.exists():
        print(f"❌ Error: {sub_wizard_path} not found!")
        raise Exception(f"Sub-wizard {script_name} not found at {sub_wizard_path}")

    result = subprocess.run([sys.executable, str(sub_wizard_path)], capture_output=False)

    if result.returncode != 0:
        raise Exception(f"Sub-wizard {script_name} failed")

    return True


def load_yaml_config(filename, directory=None):
    """Load configuration from YAML file"""
    if directory:
        path = Path(directory) / filename
    else:
        path = Path.cwd() / filename

    if path.exists():
        with open(path, 'r') as f:
            return yaml.safe_load(f)
    return {}


def main():
    print("""
╔═══════════════════════════════════════════════════════════════════════════╗
║                   SAMAPy MASTER CONFIGURATION WIZARD                      ║
║              All Parameters Match Input_Data.py Structure                 ║
╚═══════════════════════════════════════════════════════════════════════════╝
""")

    # ============================================================================
    # STEP 0: FILE DESTINATION SETUP
    # ============================================================================
    input_dir, output_dir = setup_directories()

    config = {}
    config['input_directory'] = str(input_dir)
    config['output_directory'] = str(output_dir)

    # ============================================================================
    # SECTION 1: OPTIMIZATION PARAMETERS
    # ============================================================================
    print("\n" + "=" * 80)
    print("SECTION 1: OPTIMIZATION CONFIGURATION")
    print("=" * 80)

    # Select optimization algorithm
    algo_select = {'algorithm': questionary.select("Select optimization algorithm", choices=[
        questionary.Choice('1 - PSO (Particle Swarm Optimization)', value='pso'),
        questionary.Choice('2 - ADE (Advanced Differential Evolution)', value='ade'),
        questionary.Choice('3 - GWO (Grey Wolf Optimizer)', value='gwo'),
        questionary.Choice('4 - ABC (Artificial Bee Colony)', value='abc'),
    ]).ask()}

    config['optimization_algorithm'] = algo_select['algorithm']

    # Cash flow advance is FIXED to 0 (not asked from user)
    config['Cash_Flow_adv'] = 0


    # Variable bounds
    bounds = {}
    bounds['varmin'] = questionary.text("Variable lower bounds (comma-separated, e.g., 0,0,0,0,0)", default="0,0,0,0,0").ask()
    bounds['varmax'] = questionary.text("Variable upper bounds (comma-separated, e.g., 60,60,60,20,60)", default="60,60,60,20,60").ask()

    config['VarMin'] = [float(x.strip()) for x in bounds['varmin'].split(',')]
    config['VarMax'] = [float(x.strip()) for x in bounds['varmax'].split(',')]

    # Common optimization parameters
    common_opt = {}
    common_opt['max_it'] = questionary.text("Maximum iterations (MaxIt)", default="200").ask()
    common_opt['n_pop'] = questionary.text("Population size (nPop)", default="50").ask()
    common_opt['run_time'] = questionary.text("Total number of runs (Run_Time)", default="1").ask()

    config['MaxIt'] = int(common_opt['max_it'])
    config['nPop'] = int(common_opt['n_pop'])
    config['Run_Time'] = int(common_opt['run_time'])

    # Algorithm-specific parameters
    if config['optimization_algorithm'] == 'pso':
        print("\n→ PSO Algorithm Parameters:")
        pso_params = {}
        pso_params['w'] = questionary.text("Inertia weight (w)", default="1").ask()
        pso_params['wdamp'] = questionary.text("Inertia weight damping ratio (wdamp)", default="0.99").ask()
        pso_params['c1'] = questionary.text("Personal learning coefficient (c1)", default="2").ask()
        pso_params['c2'] = questionary.text("Global learning coefficient (c2)", default="2").ask()

        config['w'] = float(pso_params['w'])
        config['wdamp'] = float(pso_params['wdamp'])
        config['c1'] = float(pso_params['c1'])
        config['c2'] = float(pso_params['c2'])

    elif config['optimization_algorithm'] == 'ade':
        print("\n→ ADE Algorithm Parameters:")
        ade_params = {}
        ade_params['f_min'] = questionary.text("Minimum scaling factor (F_min)", default="0.1").ask()
        ade_params['f_max'] = questionary.text("Maximum scaling factor (F_max)", default="0.9").ask()
        ade_params['cr_min'] = questionary.text("Minimum crossover probability (CR_min)", default="0.1").ask()
        ade_params['cr_max'] = questionary.text("Maximum crossover probability (CR_max)", default="0.9").ask()

        config['F_min'] = float(ade_params['f_min'])
        config['F_max'] = float(ade_params['f_max'])
        config['CR_min'] = float(ade_params['cr_min'])
        config['CR_max'] = float(ade_params['cr_max'])

    elif config['optimization_algorithm'] == 'abc':
        print("\n→ ABC Algorithm Parameters:")
        abc_params = {}
        abc_params['max_trials'] = questionary.text("Maximum trials (maxTrials)", default="15").ask()
        abc_params['mod_rate'] = questionary.text("Modification rate", default="0.8").ask()
        abc_params['init_radius'] = questionary.text("Initial search radius", default="0.5").ask()
        abc_params['final_radius'] = questionary.text("Final search radius", default="0.1").ask()

        config['maxTrials'] = int(abc_params['max_trials'])
        config['modification_rate'] = float(abc_params['mod_rate'])
        config['initial_search_radius'] = float(abc_params['init_radius'])
        config['final_search_radius'] = float(abc_params['final_radius'])

    elif config['optimization_algorithm'] == 'gwo':
        print("\n→ GWO Algorithm Parameters:")
        print("   (GWO uses standard parameters from common optimization settings)")

    # ============================================================================
    # SECTION 2: CALENDAR & SYSTEM LIFETIME
    # ============================================================================
    print("\n" + "=" * 80)
    print("SECTION 2: CALENDAR & SYSTEM LIFETIME")
    print("=" * 80)

    calendar = {}
    calendar['n'] = questionary.text("Lifetime of system (years)", default="25").ask()
    calendar['year'] = questionary.text("Simulation year", default="2023").ask()
    calendar['holidays'] = questionary.text("Holidays (day numbers in 365-day format, comma-separated)", default="1,51,97,100,142,182,219,248,273,282,315,359,360").ask()

    config['n'] = int(calendar['n'])
    config['year'] = int(calendar['year'])
    config['holidays'] = [int(x.strip()) for x in calendar['holidays'].split(',')]

    # ============================================================================
    # SECTION 3: LOAD DATA CONFIGURATION
    # ============================================================================
    print("\n" + "=" * 80)
    print("SECTION 3: ELECTRICAL LOAD CONFIGURATION")
    print("=" * 80)

    print("""
Load Type Options:
1 = Hourly load based on CSV file
2 = Monthly hourly average load
3 = Monthly daily average load
4 = Monthly total load
5 = Scaled generic load based on monthly total load
6 = Annual hourly average load
7 = Annual daily average load
8 = Scaled generic load based on annual total load
9 = Exactly equal to generic load
10 = Daily load profile scaled from CSV
""")

    load_config = {'load_type': questionary.select("Select electrical load input type", choices=[
        questionary.Choice('1 - Hourly load from CSV file', value=1),
        questionary.Choice('2 - Monthly hourly average', value=2),
        questionary.Choice('3 - Monthly daily average', value=3),
        questionary.Choice('4 - Monthly total load', value=4),
        questionary.Choice('5 - Generic load scaled by monthly total', value=5),
        questionary.Choice('6 - Annual hourly average', value=6),
        questionary.Choice('7 - Annual daily average', value=7),
        questionary.Choice('8 - Generic load scaled by annual total', value=8),
        questionary.Choice('9 - Exactly generic load', value=9),
        questionary.Choice('10 - Daily profile from CSV', value=10),
    ]).ask()}

    config['load_type'] = load_config['load_type']

    # Handle load type specific inputs
    if config['load_type'] == 1:
        load_csv = {'eload_csv': questionary.text("Path to Eload.csv file", default="Eload.csv").ask()}
        config['path_Eload'] = load_csv['eload_csv']
        # Copy to content folder
        copy_to_content(config['path_Eload'], 'Eload.csv')

    elif config['load_type'] == 2:
        load_monthly_h = {'monthly_h_avg': questionary.text("12 monthly hourly averages (comma-separated)", default="1,2,3,4,5,6,7,8,9,10,11,12").ask()}
        config['Monthly_haverage_load'] = [float(x.strip()) for x in load_monthly_h['monthly_h_avg'].split(',')]

    elif config['load_type'] == 3:
        load_monthly_d = {'monthly_d_avg': questionary.text("12 monthly daily averages (comma-separated)", default="10,20,31,14,15,16,17,18,19,10,11,12").ask()}
        config['Monthly_daverage_load'] = [float(x.strip()) for x in load_monthly_d['monthly_d_avg'].split(',')]

    elif config['load_type'] == 4:
        load_monthly_total = {'monthly_total': questionary.text("12 monthly total loads (comma-separated)", default="321,223,343,423,544,623,237,843,239,140,121,312").ask()}
        config['Monthly_total_load'] = [float(x.strip()) for x in load_monthly_total['monthly_total'].split(',')]

    elif config['load_type'] == 5:
        load_generic_m = {}
        load_generic_m['peak_month'] = questionary.select("Peak month for generic load", choices=['January', 'February', 'March', 'April', 'May', 'June',
                                   'July', 'August', 'September', 'October', 'November', 'December']).ask()
        load_generic_m['user_load'] = questionary.text("12 monthly total loads (comma-separated)", default="300,350,320,320,320,320,320,320,320,320,320,320").ask()
        config['peak_month'] = load_generic_m['peak_month']
        config['user_defined_load'] = [float(x.strip()) for x in load_generic_m['user_load'].split(',')]

    elif config['load_type'] == 6:
        load_annual_h = {'annual_h_avg': questionary.text("Annual hourly average load", default="1").ask()}
        config['Annual_haverage_load'] = float(load_annual_h['annual_h_avg'])

    elif config['load_type'] == 7:
        load_annual_d = {'annual_d_avg': questionary.text("Annual daily average load", default="10").ask()}
        config['Annual_daverage_load'] = float(load_annual_d['annual_d_avg'])

    elif config['load_type'] == 8:
        load_annual_total = {}
        load_annual_total['peak_month'] = questionary.select("Peak month for generic load", choices=['January', 'February', 'March', 'April', 'May', 'June',
                                   'July', 'August', 'September', 'October', 'November', 'December']).ask()
        load_annual_total['annual_total'] = questionary.text("Annual total load", default="9500").ask()
        config['peak_month'] = load_annual_total['peak_month']
        config['Annual_total_load'] = float(load_annual_total['annual_total'])

    elif config['load_type'] == 9:
        load_generic = {'peak_month': questionary.select("Peak month for generic load", choices=['January', 'February', 'March', 'April', 'May', 'June',
                                   'July', 'August', 'September', 'October', 'November', 'December']).ask()}
        config['peak_month'] = load_generic['peak_month']

    elif config['load_type'] == 10:
        load_daily = {}
        load_daily['eload_daily_csv'] = questionary.text("Path to Eload_daily.csv file", default="Eload_daily.csv").ask()
        load_daily['peak_month'] = questionary.select("Peak month for scaling", choices=['January', 'February', 'March', 'April', 'May', 'June',
                                   'July', 'August', 'September', 'October', 'November', 'December']).ask()
        config['path_Eload_daily'] = load_daily['eload_daily_csv']
        config['peak_month'] = load_daily['peak_month']
        copy_to_content(config['path_Eload_daily'], 'Eload_daily.csv')

    # Previous year load configuration
    print("\n→ Previous Year Electrical Load Configuration:")

    prev_load_config = {'load_previous_year_type': questionary.select("Select previous year load type", choices=[
        questionary.Choice('1 - Same as current year', value=1),
        questionary.Choice('2 - From CSV file', value=2),
        questionary.Choice('3 - Monthly hourly average', value=3),
        questionary.Choice('4 - Monthly daily average', value=4),
        questionary.Choice('5 - Monthly total load', value=5),
        questionary.Choice('6 - Generic scaled by monthly', value=6),
        questionary.Choice('7 - Annual hourly average', value=7),
        questionary.Choice('8 - Annual daily average', value=8),
        questionary.Choice('9 - Generic scaled by annual', value=9),
        questionary.Choice('10 - Exactly generic', value=10),
        questionary.Choice('11 - Daily profile from CSV', value=11),
    ]).ask()}

    config['load_previous_year_type'] = prev_load_config['load_previous_year_type']

    # Handle previous year load specific inputs
    if config['load_previous_year_type'] == 2:
        prev_csv = {'path': questionary.text("Path to Eload_previousyear.csv", default="Eload_previousyear.csv").ask()}
        config['path_Eload_Previous'] = prev_csv['path']
        copy_to_content(config['path_Eload_Previous'], 'Eload_previousyear.csv')

    elif config['load_previous_year_type'] == 3:
        prev_monthly_h = {'values': questionary.text("12 monthly hourly averages (comma-separated)", default="1,2,3,4,5,6,7,8,9,10,11,12").ask()}
        config['Monthly_haverage_load_previous'] = [float(x.strip()) for x in prev_monthly_h['values'].split(',')]

    elif config['load_previous_year_type'] == 4:
        prev_monthly_d = {'values': questionary.text("12 monthly daily averages (comma-separated)", default="10,20,31,14,15,16,17,18,19,10,11,12").ask()}
        config['Monthly_daverage_load_previous'] = [float(x.strip()) for x in prev_monthly_d['values'].split(',')]

    elif config['load_previous_year_type'] == 5:
        prev_monthly_total = {'values': questionary.text("12 monthly total loads (comma-separated)", default="321,223,343,423,544,623,237,843,239,140,121,312").ask()}
        config['Monthly_total_load_previous'] = [float(x.strip()) for x in prev_monthly_total['values'].split(',')]

    # Additional previous year load types 6-11 follow similar patterns...

    # ============================================================================
    # SECTION 4: THERMAL LOAD CONFIGURATION
    # ============================================================================
    print("\n" + "=" * 80)
    print("SECTION 4: THERMAL LOAD CONFIGURATION (Heating/Cooling)")
    print("=" * 80)

    print("""
Thermal Load Type Options:
1 = Hourly from Excel file (house_load.xlsx)
(Additional types can be added similar to electrical load)
""")

    thermal_config = {'tload_type': questionary.select("Select thermal load input type", choices=[
        questionary.Choice('1 - Hourly from house_load.xlsx', value=1),
    ]).ask()}

    config['Tload_type'] = thermal_config['tload_type']

    if config['Tload_type'] == 1:
        tload_file = {'path': questionary.text("Path to house_load.xlsx file", default="house_load.xlsx").ask()}
        config['path_house_load'] = tload_file['path']
        copy_to_content(config['path_house_load'], 'house_load.xlsx')
        print("   Note: File should contain columns for heating load (Hload) and cooling load (Cload)")
        print("   Note: Tload (total thermal load) will be calculated as Hload + Cload for NG pricing")

    # ============================================================================
    # SECTION 5: WEATHER DATA CONFIGURATION (kept for legacy, moved to section 6)
    # ============================================================================

    # ============================================================================
    # SECTION 6: IRRADIANCE (SOLAR) CONFIGURATION
    # ============================================================================
    print("\n" + "=" * 80)
    print("SECTION 6: SOLAR IRRADIANCE CONFIGURATION")
    print("=" * 80)

    print("""
Irradiance Input Options:
1 = Hourly irradiance based on POA calculator (requires METEO.csv)
2 = Hourly POA irradiance from user CSV file
""")

    irradiance_config = {}
    irradiance_config['g_type'] = questionary.select("Select irradiance input method", choices=[
        questionary.Choice('1 - POA calculator from METEO.csv', value=1),
        questionary.Choice('2 - User CSV file (Irradiance.csv)', value=2),
    ]).ask()
    irradiance_config['azimuth'] = questionary.text("PV azimuth angle (degrees, 180=south)", default="180").ask()
    irradiance_config['tilt'] = questionary.text("PV tilt angle (degrees)", default="33").ask()
    irradiance_config['soiling'] = questionary.text("Soiling losses (%)", default="5").ask()

    config['G_type'] = irradiance_config['g_type']
    config['azimuth'] = float(irradiance_config['azimuth'])
    config['tilt'] = float(irradiance_config['tilt'])
    config['soiling'] = float(irradiance_config['soiling'])

    if config['G_type'] == 1:
        meteo = {'path': questionary.text("Path to METEO.csv file", default="METEO.csv").ask()}
        config['weather_url'] = meteo['path']
        copy_to_content(config['weather_url'], 'METEO.csv')
    else:
        irrad_csv = {'path': questionary.text("Path to Irradiance.csv file", default="Irradiance.csv").ask()}
        config['path_G'] = irrad_csv['path']
        copy_to_content(config['path_G'], 'Irradiance.csv')

    # ============================================================================
    # SECTION 6B: TEMPERATURE CONFIGURATION
    # ============================================================================
    print("\n→ Temperature Configuration:")
    print("""
Temperature Input Options:
1 = From METEO.csv (NSEDB)
2 = User CSV file
3 = Monthly average temperature
4 = Annual average temperature
""")

    temp_config = {'t_type': questionary.select("Select temperature input method", choices=[
        questionary.Choice('1 - From METEO.csv', value=1),
        questionary.Choice('2 - User CSV file', value=2),
        questionary.Choice('3 - Monthly average', value=3),
        questionary.Choice('4 - Annual average', value=4),
    ]).ask()}

    config['T_type'] = temp_config['t_type']

    if config['T_type'] == 2:
        temp_csv = {'path': questionary.text("Path to Temperature.csv", default="Temperature.csv").ask()}
        config['path_T'] = temp_csv['path']
        copy_to_content(config['path_T'], 'Temperature.csv')
    elif config['T_type'] == 3:
        temp_monthly = {'values': questionary.text("12 monthly average temperatures (°C, comma-separated)", default="-2,-5,-2,1,3,6,15,22,27,23,16,7").ask()}
        config['Monthly_average_temperature'] = [float(x.strip()) for x in temp_monthly['values'].split(',')]
    elif config['T_type'] == 4:
        temp_annual = {'value': questionary.text("Annual average temperature (°C)", default="12").ask()}
        config['Annual_average_temperature'] = float(temp_annual['value'])

    # ============================================================================
    # SECTION 6C: WIND SPEED CONFIGURATION
    # ============================================================================
    print("\n→ Wind Speed Configuration:")
    print("""
Wind Speed Input Options:
1 = From METEO.csv (NSEDB)
2 = User CSV file
3 = Monthly average wind speed
4 = Annual average wind speed
""")

    wind_config = {'ws_type': questionary.select("Select wind speed input method", choices=[
        questionary.Choice('1 - From METEO.csv', value=1),
        questionary.Choice('2 - User CSV file', value=2),
        questionary.Choice('3 - Monthly average', value=3),
        questionary.Choice('4 - Annual average', value=4),
    ]).ask()}

    config['WS_type'] = wind_config['ws_type']

    if config['WS_type'] == 2:
        wind_csv = {'path': questionary.text("Path to WSPEED.csv", default="WSPEED.csv").ask()}
        config['path_WS'] = wind_csv['path']
        copy_to_content(config['path_WS'], 'WSPEED.csv')
    elif config['WS_type'] == 3:
        wind_monthly = {'values': questionary.text("12 monthly average wind speeds (m/s, comma-separated)", default="14.1,21,12.2,31,12.2,11.2,12.1,13,21,9.2,12.3,18.1").ask()}
        config['Monthly_average_windspeed'] = [float(x.strip()) for x in wind_monthly['values'].split(',')]
    elif config['WS_type'] == 4:
        wind_annual = {'value': questionary.text("Annual average wind speed (m/s)", default="10").ask()}
        config['Annual_average_windspeed'] = float(wind_annual['value'])

    # ============================================================================
    # SECTION 7: SYSTEM CONSTRAINTS
    # ============================================================================
    print("\n" + "=" * 80)
    print("SECTION 7: SYSTEM CONSTRAINTS & REQUIREMENTS")
    print("=" * 80)

    constraints = {}
    constraints['lpsp_max'] = questionary.text("Maximum loss of power supply probability (%)", default="0.0999").ask()
    constraints['re_min'] = questionary.text("Minimum renewable energy capacity (%)", default="50").ask()
    constraints['em'] = questionary.select("Emission mode", choices=[
        questionary.Choice('0 - NPC only', value=0),
        questionary.Choice('1 - NPC + LEM (lifecycle emissions)', value=1),
    ]).ask()

    config['LPSP_max_rate'] = float(constraints['lpsp_max'])
    config['LPSP_max'] = config['LPSP_max_rate'] / 100
    config['RE_min_rate'] = float(constraints['re_min'])
    config['RE_min'] = config['RE_min_rate'] / 100
    config['EM'] = constraints['em']

    # ============================================================================
    # SECTION 8: COMPONENT SELECTION
    # ============================================================================
    print("\n" + "=" * 80)
    print("SECTION 8: SYSTEM COMPONENTS SELECTION")
    print("=" * 80)

    print("""
Select which components to include in your hybrid energy system:
- PV (Solar Photovoltaic)
- WT (Wind Turbine)
- DG (Diesel Generator)
- Battery Storage (Lead-acid or Li-ion)
- Grid Connection
- Heat Pump
- Electric Vehicle (EV)
""")

    components = {}
    components['pv'] = questionary.confirm("Include PV (Solar)?", default=True).ask()
    components['wt'] = questionary.confirm("Include WT (Wind Turbine)?", default=False).ask()
    components['dg'] = questionary.confirm("Include DG (Diesel Generator)?", default=False).ask()
    components['bat'] = questionary.confirm("Include Battery Storage?", default=True).ask()
    components['grid'] = questionary.confirm("Include Grid Connection?", default=True).ask()
    components['hp'] = questionary.confirm("Include Heat Pump?", default=False).ask()
    components['ev'] = questionary.confirm("Include Electric Vehicle?", default=False).ask()

    config['PV'] = 1 if components['pv'] else 0
    config['WT'] = 1 if components['wt'] else 0
    config['DG'] = 1 if components['dg'] else 0
    config['Bat'] = 1 if components['bat'] else 0
    config['Grid'] = 1 if components['grid'] else 0
    config['HP'] = 1 if components['hp'] else 0
    config['EV'] = 1 if components['ev'] else 0

    # Battery type selection
    if config['Bat']:
        bat_type = {'type': questionary.select("Select battery type", choices=[
            questionary.Choice('Lead-acid', value='lead_acid'),
            questionary.Choice('Li-ion', value='li_ion'),
        ]).ask()}
        config['Lead_acid'] = 1 if bat_type['type'] == 'lead_acid' else 0
        config['Li_ion'] = 1 if bat_type['type'] == 'li_ion' else 0
    else:
        config['Lead_acid'] = 0
        config['Li_ion'] = 0

    # Net metering and capacity options
    if config['Grid']:
        grid_opts = {}
        grid_opts['nem'] = questionary.confirm("Enable Net Energy Metering (NEM)?", default=True).ask()
        grid_opts['cap_option'] = questionary.select("System sizing constraint", choices=[
            questionary.Choice('1 - NEM cap (specify kW limit)', value=1),
            questionary.Choice('2 - Size to recent annual load', value=2),
            questionary.Choice('3 - Size to rooftop area', value=3),
            questionary.Choice('4 - No limit', value=4),
        ]).ask()
        config['NEM'] = 1 if grid_opts['nem'] else 0

        config['cap_option'] = grid_opts['cap_option']

        if config['cap_option'] == 1:
            cap = {'size': questionary.text("NEM capacity cap (kW)", default="0").ask()}
            config['cap_size'] = float(cap['size'])
        elif config['cap_option'] == 2:
            gen_cap = {'percent': questionary.text("Generation capacity % of consumption", default="150").ask()}
            config['generation_cap'] = float(gen_cap['percent'])
        elif config['cap_option'] == 3:
            roof = {}
            roof['area'] = questionary.text("Available roof surface (m²)", default="40").ask()
            roof['ratio'] = questionary.text("PV surface per kW (m²/kW)", default="5").ask()
            config['available_roof_surface'] = float(roof['area'])
            config['PVPanel_surface_per_rated_capacity'] = float(roof['ratio'])

        # NEM fee
        nem_fee = {'fee': questionary.text("NEM one-time setup fee ($)", default="0").ask()}
        config['NEM_fee'] = float(nem_fee['fee'])
    else:
        config['NEM'] = 0
        config['cap_option'] = 4

    # NG_Grid is FIXED - automatically set based on HP
    #config['NG_Grid'] = 1 if config['HP'] == 1 else 0


    # ============================================================================
    # SECTION 9: ECONOMIC & FINANCIAL PARAMETERS
    # ============================================================================
    print("\n" + "=" * 80)
    print("SECTION 9: ECONOMIC & FINANCIAL PARAMETERS")
    print("=" * 80)

    financial = {}
    financial['nominal_ir'] = questionary.text("Nominal discount rate (%)", default="2.75").ask()
    financial['inflation'] = questionary.text("Expected inflation rate (%)", default="2").ask()
    financial['budget'] = questionary.text("Budget limit on total capital cost ($)", default="2000000").ask()
    financial['tax_rate'] = questionary.text("Equipment sale tax (%)", default="0").ask()
    financial['re_incentives'] = questionary.text("Federal tax credit/RE incentives (%)", default="30").ask()

    config['n_ir_rate'] = float(financial['nominal_ir'])
    config['n_ir'] = config['n_ir_rate'] / 100
    config['e_ir_rate'] = float(financial['inflation'])
    config['e_ir'] = config['e_ir_rate'] / 100
    config['ir'] = (config['n_ir'] - config['e_ir']) / (1 + config['e_ir'])
    config['Budget'] = float(financial['budget'])
    config['Tax_rate'] = float(financial['tax_rate'])
    config['System_Tax'] = config['Tax_rate'] / 100
    config['RE_incentives_rate'] = float(financial['re_incentives'])
    config['RE_incentives'] = config['RE_incentives_rate'] / 100

    # ============================================================================
    # SECTION 10: PRICING METHOD SELECTION
    # ============================================================================
    print("\n" + "=" * 80)
    print("SECTION 10: PRICING METHOD SELECTION")
    print("=" * 80)

    pricing = {'method': questionary.select("Select pricing calculation method", choices=[
        questionary.Choice('1 - Top-down (NREL percentages)', value=1),
        questionary.Choice('2 - Bottom-up (detailed components)', value=2),
    ]).ask()}

    config['Pricing_method'] = pricing['method']

    # ============================================================================
    # SECTION 11: PV CONFIGURATION
    # ============================================================================
    if config['PV']:
        print("\n" + "=" * 80)
        print("SECTION 11: PHOTOVOLTAIC (PV) CONFIGURATION")
        print("=" * 80)

        if config['Pricing_method'] == 1:
            # Top-down pricing
            print("\n→ Using Top-Down Pricing (NREL percentages)")
            pv_topdown = {'total_price': questionary.text("Total PV system price ($/kW)", default="2750").ask()}

            Total_PV_price = float(pv_topdown['total_price'])
            config['Total_PV_price'] = Total_PV_price

            from samapy.utilities.top_down import top_down_pricing
            # Calculate using NREL percentages
            Engineering_Costs, C_PV, R_PV, C_I, R_I = top_down_pricing(Total_PV_price)

            config['Engineering_Costs'] = float(Engineering_Costs)
            config['C_PV'] = float(C_PV)
            config['R_PV'] = float(R_PV)
            config['C_I'] = float(C_I)
            config['R_I'] = float(R_I)

            print(f"\n✓ Calculated costs:")
            print(f"  Engineering: ${Engineering_Costs:.2f}/kW")
            print(f"  PV Module: ${C_PV:.2f}/kW")
            print(f"  Inverter: ${C_I:.2f}/kW")

        else:
            # Bottom-up pricing
            print("\n→ Using Bottom-Up Pricing (detailed components)")

            # Engineering costs breakdown
            eng_costs = {}
            eng_costs['fieldwork'] = questionary.text("Fieldwork costs ($/kW)", default="178").ask()
            eng_costs['officework'] = questionary.text("Officework costs ($/kW)", default="696").ask()
            eng_costs['other'] = questionary.text("Other costs ($/kW)", default="586").ask()
            eng_costs['permitting'] = questionary.text("Permitting & Inspection ($/kW)", default="0").ask()
            eng_costs['electrical_bos'] = questionary.text("Electrical BoS ($/kW)", default="333").ask()
            eng_costs['structural_bos'] = questionary.text("Structural BoS ($/kW)", default="237").ask()
            eng_costs['supply_chain'] = questionary.text("Supply chain costs ($/kW)", default="0").ask()
            eng_costs['profit'] = questionary.text("Profit costs ($/kW)", default="0").ask()
            eng_costs['sales_tax'] = questionary.text("Sales tax ($/kW)", default="0").ask()

            config['Fieldwork'] = float(eng_costs['fieldwork'])
            config['Officework'] = float(eng_costs['officework'])
            config['Other'] = float(eng_costs['other'])
            config['Permiting_and_Inspection'] = float(eng_costs['permitting'])
            config['Electrical_BoS'] = float(eng_costs['electrical_bos'])
            config['Structrual_BoS'] = float(eng_costs['structural_bos'])
            config['Supply_Chain_costs'] = float(eng_costs['supply_chain'])
            config['Profit_costs'] = float(eng_costs['profit'])
            config['Sales_tax'] = float(eng_costs['sales_tax'])

            config['Engineering_Costs'] = (config['Sales_tax'] + config['Profit_costs'] +
                                           config['Fieldwork'] + config['Officework'] +
                                           config['Other'] + config['Permiting_and_Inspection'] +
                                           config['Electrical_BoS'] + config['Structrual_BoS'] +
                                           config['Supply_Chain_costs'])

            pv_bottomup = {}
            pv_bottomup['c_pv'] = questionary.text("PV module cost ($/kW)", default="338").ask()
            pv_bottomup['r_pv'] = questionary.text("PV replacement cost ($/kW)", default="338").ask()
            pv_bottomup['c_i'] = questionary.text("Inverter cost ($/kW)", default="314").ask()
            pv_bottomup['r_i'] = questionary.text("Inverter replacement cost ($/kW)", default="314").ask()
            pv_bottomup['mo_pv'] = questionary.text("PV O&M cost ($/kW/year)", default="30.36").ask()
            pv_bottomup['mo_i'] = questionary.text("Inverter O&M cost ($/kW/year)", default="0").ask()

            config['C_PV'] = float(pv_bottomup['c_pv'])
            config['R_PV'] = float(pv_bottomup['r_pv'])
            config['C_I'] = float(pv_bottomup['c_i'])
            config['R_I'] = float(pv_bottomup['r_i'])
            config['MO_PV'] = float(pv_bottomup['mo_pv'])
            config['MO_I'] = float(pv_bottomup['mo_i'])

        # Common PV technical parameters - CRITICAL MISSING PARAMETERS
        pv_technical = {}
        pv_technical['fpv'] = questionary.text("PV derating factor", default="0.9").ask()
        pv_technical['tcof'] = questionary.text("Temperature coefficient (%/°C)", default="-0.3").ask()
        pv_technical['tref'] = questionary.text("Reference temperature (°C)", default="25").ask()
        pv_technical['tc_noct'] = questionary.text("NOCT temperature (°C)", default="45").ask()
        pv_technical['ta_noct'] = questionary.text("Ambient temp at NOCT (°C)", default="20").ask()
        pv_technical['g_noct'] = questionary.text("Irradiance at NOCT (W/m²)", default="800").ask()
        pv_technical['gama'] = questionary.text("Gamma factor", default="0.9").ask()
        pv_technical['n_pv'] = questionary.text("PV module efficiency", default="0.2182").ask()
        pv_technical['gref'] = questionary.text("Reference irradiance (W/m²)", default="1000").ask()
        pv_technical['l_pv'] = questionary.text("PV lifetime (years)", default="25").ask()

        config['fpv'] = float(pv_technical['fpv'])
        config['Tcof'] = float(pv_technical['tcof'])
        config['Tref'] = float(pv_technical['tref'])
        config['Tc_noct'] = float(pv_technical['tc_noct'])
        config['Ta_noct'] = float(pv_technical['ta_noct'])
        config['G_noct'] = float(pv_technical['g_noct'])
        config['gama'] = float(pv_technical['gama'])
        config['n_PV'] = float(pv_technical['n_pv'])
        config['Gref'] = float(pv_technical['gref'])
        config['L_PV'] = int(pv_technical['l_pv'])

        # Calculate replacement time
        config['RT_PV'] = max(0, int(np.ceil(config['n'] / config['L_PV']) - 1))

        # Inverter parameters
        inv_params = {}
        inv_params['n_i'] = questionary.text("Inverter efficiency", default="0.96").ask()
        inv_params['dc_ac_ratio'] = questionary.text("Maximum DC/AC ratio", default="1.99").ask()
        inv_params['l_i'] = questionary.text("Inverter lifetime (years)", default="25").ask()

        config['n_I'] = float(inv_params['n_i'])
        config['DC_AC_ratio'] = float(inv_params['dc_ac_ratio'])
        config['L_I'] = int(inv_params['l_i'])
        config['RT_I'] = max(0, int(np.ceil(config['n'] / config['L_I']) - 1))

        # Rated capacity
        pv_rated = {'ppv_r': questionary.text("PV module rated power (kW)", default="1").ask()}
        config['Ppv_r'] = float(pv_rated['ppv_r'])

    # ============================================================================
    # SECTION 12: WIND TURBINE CONFIGURATION
    # ============================================================================
    if config['WT']:
        print("\n" + "=" * 80)
        print("SECTION 12: WIND TURBINE CONFIGURATION")
        print("=" * 80)

        wt_params = {}
        wt_params['c_wt'] = questionary.text("WT capital cost ($/kW)", default="1200").ask()
        wt_params['r_wt'] = questionary.text("WT replacement cost ($/kW)", default="1200").ask()
        wt_params['mo_wt'] = questionary.text("WT O&M cost ($/kW/year)", default="40").ask()
        wt_params['lifetime'] = questionary.text("WT lifetime (years)", default="20").ask()
        wt_params['h_hub'] = questionary.text("Hub height (m)", default="17").ask()
        wt_params['h0'] = questionary.text("Anemometer height (m)", default="43.6").ask()
        wt_params['nw'] = questionary.text("Electrical efficiency", default="1").ask()
        wt_params['v_cut_in'] = questionary.text("Cut-in wind speed (m/s)", default="2.5").ask()
        wt_params['v_rated'] = questionary.text("Rated wind speed (m/s)", default="9.5").ask()
        wt_params['v_cut_out'] = questionary.text("Cut-out wind speed (m/s)", default="25").ask()
        wt_params['alfa'] = questionary.text("Coefficient of friction", default="0.14").ask()

        config['C_WT'] = float(wt_params['c_wt'])
        config['R_WT'] = float(wt_params['r_wt'])
        config['MO_WT'] = float(wt_params['mo_wt'])
        config['L_WT'] = int(wt_params['lifetime'])
        config['RT_WT'] = max(0, int(np.ceil(config['n'] / config['L_WT']) - 1))
        config['h_hub'] = float(wt_params['h_hub'])
        config['h0'] = float(wt_params['h0'])
        config['nw'] = float(wt_params['nw'])
        config['v_cut_in'] = float(wt_params['v_cut_in'])
        config['v_rated'] = float(wt_params['v_rated'])
        config['v_cut_out'] = float(wt_params['v_cut_out'])
        config['alfa_wind_turbine'] = float(wt_params['alfa'])

        # Rated capacity
        wt_rated = {'pwt_r': questionary.text("WT rated power (kW)", default="1").ask()}
        config['Pwt_r'] = float(wt_rated['pwt_r'])

    # ============================================================================
    # SECTION 13: DIESEL GENERATOR CONFIGURATION
    # ============================================================================
    if config['DG']:
        print("\n" + "=" * 80)
        print("SECTION 13: DIESEL GENERATOR CONFIGURATION")
        print("=" * 80)

        dg_params = {}
        dg_params['c_dg'] = questionary.text("DG capital cost ($/kW)", default="818").ask()
        dg_params['r_dg'] = questionary.text("DG replacement cost ($/kW)", default="818").ask()
        dg_params['mo_dg'] = questionary.text("DG O&M cost ($/operating hour)", default="0.016").ask()
        dg_params['lifetime'] = questionary.text("DG lifetime (hours)", default="24000").ask()
        dg_params['fuel_cost'] = questionary.text("Fuel cost ($/L)", default="1.281").ask()
        dg_params['fuel_adj_rate'] = questionary.text("Fuel cost escalation rate (%/year)", default="2").ask()
        dg_params['fuel_a'] = questionary.text("Fuel curve coefficient a (L/hr/kW output)", default="0.4388").ask()
        dg_params['fuel_b'] = questionary.text("Fuel curve coefficient b (L/hr/kW rated)", default="0.1097").ask()
        dg_params['min_load'] = questionary.text("Minimum load ratio", default="0.25").ask()

        config['C_DG'] = float(dg_params['c_dg'])
        config['R_DG'] = float(dg_params['r_dg'])
        config['MO_DG'] = float(dg_params['mo_dg'])
        config['TL_DG'] = int(dg_params['lifetime'])
        config['C_fuel'] = float(dg_params['fuel_cost'])
        config['C_fuel_adj_rate'] = float(dg_params['fuel_adj_rate'])
        config['C_fuel_adj'] = config['C_fuel_adj_rate'] / 100
        config['a'] = float(dg_params['fuel_a'])
        config['b'] = float(dg_params['fuel_b'])
        config['LR_DG'] = float(dg_params['min_load'])

        # DG Emissions
        print("\n→ Diesel Generator Emissions:")
        dg_emissions = {}
        dg_emissions['co2'] = questionary.text("CO2 emissions (kg/L)", default="2.29").ask()
        dg_emissions['co'] = questionary.text("CO emissions (kg/L)", default="0").ask()
        dg_emissions['nox'] = questionary.text("NOx emissions (kg/L)", default="0").ask()
        dg_emissions['so2'] = questionary.text("SO2 emissions (kg/L)", default="0").ask()

        config['CO2'] = float(dg_emissions['co2'])
        config['CO'] = float(dg_emissions['co'])
        config['NOx'] = float(dg_emissions['nox'])
        config['SO2'] = float(dg_emissions['so2'])

        # Rated capacity
        dg_rated = {'cdg_r': questionary.text("DG rated capacity (kW)", default="5.5").ask()}
        config['Cdg_r'] = float(dg_rated['cdg_r'])

    # ============================================================================
    # SECTION 14: BATTERY STORAGE CONFIGURATION
    # ============================================================================
    if config['Bat']:
        print("\n" + "=" * 80)
        print("SECTION 14: BATTERY STORAGE CONFIGURATION")
        print("=" * 80)

        bat_params = {}
        bat_params['c_bat'] = questionary.text("Battery capital cost ($/kWh)", default="1450").ask()
        bat_params['r_bat'] = questionary.text("Battery replacement cost ($/kWh)", default="1450").ask()
        bat_params['mo_bat'] = questionary.text("Battery O&M cost ($/kWh/year)", default="10").ask()
        bat_params['soc_min'] = questionary.text("Minimum state of charge (SOC_min)", default="0.1").ask()
        bat_params['soc_max'] = questionary.text("Maximum state of charge (SOC_max)", default="1").ask()
        bat_params['soc_init'] = questionary.text("Initial state of charge (SOC_initial)", default="0.5").ask()
        bat_params['self_discharge'] = questionary.text("Hourly self-discharge rate", default="0").ask()
        bat_params['lifetime'] = questionary.text("Battery lifetime (years)", default="10").ask()

        config['C_B'] = float(bat_params['c_bat'])
        config['R_B'] = float(bat_params['r_bat'])
        config['MO_B'] = float(bat_params['mo_bat'])
        config['SOC_min'] = float(bat_params['soc_min'])
        config['SOC_max'] = float(bat_params['soc_max'])
        config['SOC_initial'] = float(bat_params['soc_init'])
        config['self_discharge_rate'] = float(bat_params['self_discharge'])
        config['L_B'] = int(bat_params['lifetime'])
        config['RT_B'] = max(0, int(np.ceil(config['n'] / config['L_B']) - 1))

        # Battery type specific parameters
        if config.get('Lead_acid', 0) == 1:
            print("\n→ Lead-Acid Battery Parameters:")
            lead_acid = {}
            lead_acid['cnom'] = questionary.text("Nominal capacity (Ah)", default="83.4").ask()
            lead_acid['alfa'] = questionary.text("Maximum charge rate (A/Ah)", default="1").ask()
            lead_acid['c'] = questionary.text("Capacity ratio", default="0.403").ask()
            lead_acid['k'] = questionary.text("Rate constant (1/h)", default="0.827").ask()
            lead_acid['ich_max'] = questionary.text("Maximum charge current (A)", default="16.7").ask()
            lead_acid['vnom'] = questionary.text("Nominal voltage (V)", default="12").ask()
            lead_acid['ef_bat'] = questionary.text("Round trip efficiency", default="0.8").ask()
            lead_acid['q_lifetime'] = questionary.text("Lifetime throughput (kWh)", default="8000").ask()

            config['Cnom_Leadacid'] = float(lead_acid['cnom'])
            config['alfa_battery_leadacid'] = float(lead_acid['alfa'])
            config['c'] = float(lead_acid['c'])
            config['k_lead_acid'] = float(lead_acid['k'])
            config['Ich_max_leadacid'] = float(lead_acid['ich_max'])
            config['Vnom_leadacid'] = float(lead_acid['vnom'])
            config['ef_bat_leadacid'] = float(lead_acid['ef_bat'])
            config['Q_lifetime_leadacid'] = float(lead_acid['q_lifetime'])

            # Rated capacity for Lead-acid
            config['Cbt_r'] = (config['Vnom_leadacid'] * config['Cnom_Leadacid']) / 1000

        if config.get('Li_ion', 0) == 1:
            print("\n→ Li-ion Battery Parameters:")
            li_ion = {}
            li_ion['cnom'] = questionary.text("Nominal capacity (Ah)", default="167").ask()
            li_ion['ich_max'] = questionary.text("Maximum charge current (A)", default="167").ask()
            li_ion['idch_max'] = questionary.text("Maximum discharge current (A)", default="500").ask()
            li_ion['alfa'] = questionary.text("Maximum charge rate (A/Ah)", default="1").ask()
            li_ion['vnom'] = questionary.text("Nominal voltage (V)", default="6").ask()
            li_ion['ef_bat'] = questionary.text("Round trip efficiency", default="0.90").ask()
            li_ion['q_lifetime'] = questionary.text("Lifetime throughput (kWh)", default="3000").ask()

            config['Cnom_Li'] = float(li_ion['cnom'])
            config['Ich_max_Li_ion'] = float(li_ion['ich_max'])
            config['Idch_max_Li_ion'] = float(li_ion['idch_max'])
            config['alfa_battery_Li_ion'] = float(li_ion['alfa'])
            config['Vnom_Li_ion'] = float(li_ion['vnom'])
            config['ef_bat_Li'] = float(li_ion['ef_bat'])
            config['Q_lifetime_Li'] = float(li_ion['q_lifetime'])

            # Rated capacity for Li-ion
            config['Cbt_r'] = (config['Vnom_Li_ion'] * config['Cnom_Li']) / 1000

        # Charger parameters
        print("\n→ Battery Charger Parameters:")
        charger = {}
        charger['c_ch'] = questionary.text("Charger capital cost ($)", default="0").ask()
        charger['r_ch'] = questionary.text("Charger replacement cost ($)", default="0").ask()
        charger['mo_ch'] = questionary.text("Charger O&M cost ($/year)", default="0").ask()
        charger['l_ch'] = questionary.text("Charger lifetime (years)", default="25").ask()

        config['C_CH'] = float(charger['c_ch'])
        config['R_CH'] = float(charger['r_ch'])
        config['MO_CH'] = float(charger['mo_ch'])
        config['L_CH'] = int(charger['l_ch'])
        config['RT_CH'] = max(0, int(np.ceil(config['n'] / config['L_CH']) - 1))

    # ============================================================================
    # SECTION 15: HEAT PUMP CONFIGURATION
    # ============================================================================
    if config['HP']:
        print("\n" + "=" * 80)
        print("SECTION 15: HEAT PUMP CONFIGURATION")
        print("=" * 80)

        hp_selection = {'hp_brand': questionary.select("Select heat pump brand/model", choices=[
            questionary.Choice('Bosch', value='Bosch'),
            questionary.Choice('Goodman', value='Goodman'),
        ]).ask()}

        config['HP_brand'] = hp_selection['hp_brand']

        hp_params = {}
        hp_params['c_hp'] = questionary.text("HP capital cost ($/HP rated size)", default="109.5").ask()
        hp_params['r_hp'] = questionary.text("HP replacement cost ($/HP rated size)", default="109.5").ask()
        hp_params['mo_hp'] = questionary.text("HP O&M cost ($/year)", default="20").ask()
        hp_params['lifetime'] = questionary.text("HP lifetime (years)", default="5").ask()

        config['C_HP'] = float(hp_params['c_hp'])
        config['R_HP'] = float(hp_params['r_hp'])
        config['MO_HP'] = float(hp_params['mo_hp'])
        config['L_HP'] = int(hp_params['lifetime'])
        config['RT_HP'] = max(0, int(np.ceil(config['n'] / config['L_HP']) - 1))

        # Rated capacity
        hp_rated = {'php_r': questionary.text("HP rated size (BTU/hr)", default="1000").ask()}
        config['Php_r'] = float(hp_rated['php_r'])

    # ============================================================================
    # SECTION 16: ELECTRIC VEHICLE CONFIGURATION (ALL PARAMETERS)
    # ============================================================================
    if config['EV']:
        print("\n" + "=" * 80)
        print("SECTION 16: ELECTRIC VEHICLE CONFIGURATION")
        print("=" * 80)

        ev_basic = {}
        ev_basic['c_ev'] = questionary.text("EV battery capacity (kWh)", default="82").ask()
        ev_basic['soce_min'] = questionary.text("Minimum SOC", default="0.03").ask()
        ev_basic['soce_max'] = questionary.text("Maximum SOC", default="0.97").ask()
        ev_basic['soce_init'] = questionary.text("Initial SOC", default="0.85").ask()
        ev_basic['pev_max'] = questionary.text("Maximum charge/discharge rate (kW)", default="11").ask()
        ev_basic['range_ev'] = questionary.text("EV range with full charge (km)", default="468").ask()
        ev_basic['daily_trip'] = questionary.text("Average daily travel distance (km)", default="68").ask()
        ev_basic['soc_dep'] = questionary.text("SOC at departure time", default="0.85").ask()
        ev_basic['n_e'] = questionary.text("EV battery charge efficiency", default="0.9").ask()
        ev_basic['self_discharge_ev'] = questionary.text("EV battery self-discharge rate", default="0").ask()

        config['C_ev'] = float(ev_basic['c_ev'])
        config['SOCe_min'] = float(ev_basic['soce_min'])
        config['SOCe_max'] = float(ev_basic['soce_max'])
        config['C_ev_usable'] = config['C_ev'] * (config['SOCe_max'] - config['SOCe_min'])
        config['SOCe_initial'] = float(ev_basic['soce_init'])
        config['Pev_max'] = float(ev_basic['pev_max'])
        config['Range_EV'] = float(ev_basic['range_ev'])
        config['Daily_trip'] = float(ev_basic['daily_trip'])
        config['SOC_dep'] = float(ev_basic['soc_dep'])
        config['SOC_arr'] = config['SOC_dep'] - (
                    (config['Daily_trip'] * config['C_ev_usable']) / (config['Range_EV'] * config['C_ev']))
        config['n_e'] = float(ev_basic['n_e'])
        config['self_discharge_rate_ev'] = float(ev_basic['self_discharge_ev'])

        # EV lifetime and degradation
        ev_lifetime = {}
        ev_lifetime['l_ev_dis'] = questionary.text("EV battery lifetime (km)", default="400000").ask()
        ev_lifetime['degradation'] = questionary.text("Degradation percent per step", default="0.019").ask()
        ev_lifetime['step_km'] = questionary.text("Step distance (km)", default="1000").ask()
        ev_lifetime['l_ev_years'] = questionary.text("EV lifetime (years)", default="25").ask()

        config['L_EV_dis'] = float(ev_lifetime['l_ev_dis'])
        config['degradation_percent'] = float(ev_lifetime['degradation'])
        config['step_km'] = float(ev_lifetime['step_km'])
        config['L_EV'] = int(ev_lifetime['l_ev_years'])
        config['RT_EV'] = max(0, int(np.ceil(config['n'] / config['L_EV']) - 1))

        # Note: Q_lifetime_ev (EV battery lifetime throughput) will be calculated automatically
        # using calculate_ev_battery_throughput() with the parameters above

        # EV timing parameters
        ev_timing = {}
        ev_timing['tin'] = questionary.text("Arrival time to home (hour, 0-23)", default="17").ask()
        ev_timing['tout'] = questionary.text("Departure time from home (hour, 0-23)", default="8").ask()
        ev_timing['treat_special_home'] = questionary.confirm("Treat weekends/holidays as home days?", default=False).ask()

        config['Tin'] = int(ev_timing['tin'])
        config['Tout'] = int(ev_timing['tout'])
        config['treat_special_days_as_home'] = ev_timing['treat_special_home']

        # EV costs
        ev_costs = {}
        ev_costs['cost_ev'] = questionary.text("EV charger/infrastructure cost ($)", default="0").ask()
        ev_costs['r_evb'] = questionary.text("EV battery replacement cost ($)", default="27000").ask()
        ev_costs['mo_ev'] = questionary.text("EV O&M cost ($/year)", default="0").ask()

        config['Cost_EV'] = float(ev_costs['cost_ev'])
        config['R_EVB'] = float(ev_costs['r_evb'])
        config['MO_EV'] = float(ev_costs['mo_ev'])

        print("\n✓ EV configuration complete (all Input_Data.py parameters included)")

    # ============================================================================
    # SECTION 17: ELECTRICITY RATE STRUCTURES
    # ============================================================================
    # Off-grid: ask if user wants to compare with grid-only scenario
    config['compare_with_grid'] = 0
    if not config['Grid']:
        print("\n" + "=" * 80)
        print("SECTION 17: GRID COMPARISON (Off-Grid System)")
        print("=" * 80)
        offgrid_cmp = {'compare': questionary.confirm("Compare your off-grid HES with a grid-only system? (enables electricity cost comparison charts)", default=True).ask()}
        config['compare_with_grid'] = 1 if offgrid_cmp['compare'] else 0

    if config['Grid'] or config['compare_with_grid']:
        print("\n" + "=" * 80)
        print("SECTION 17: ELECTRICITY RATE STRUCTURES")
        if not config['Grid']:
            print("  (Rates used for grid-only comparison scenario)")
        print("=" * 80)

        # Grid economics - fixed charges
        grid_fixed = {}
        grid_fixed['annual_exp'] = questionary.text("Annual grid expenses ($)", default="0").ask()
        grid_fixed['sale_tax'] = questionary.text("Electricity sale tax rate (%)", default="13").ask()
        grid_fixed['grid_tax'] = questionary.text("Grid tax amount ($/kWh)", default="0.0353").ask()
        grid_fixed['credit'] = questionary.text("Grid credits ($)", default="0").ask()

        config['Annual_expenses'] = float(grid_fixed['annual_exp'])
        config['Grid_sale_tax_rate'] = float(grid_fixed['sale_tax'])
        config['Grid_Tax'] = config['Grid_sale_tax_rate'] / 100
        config['Grid_Tax_amount'] = float(grid_fixed['grid_tax'])
        config['Grid_credit'] = float(grid_fixed['credit'])

        # Grid escalation
        grid_esc = {'projection': questionary.select("Grid electricity escalation projection", choices=[
            questionary.Choice('1 - Flat yearly rate', value=1),
            questionary.Choice('2 - 25 different yearly rates', value=2),
        ]).ask()}

        config['Grid_escalation_projection'] = grid_esc['projection']

        if grid_esc['projection'] == 1:
            grid_esc_rate = {'rate': questionary.text("Yearly escalation rate (%)", default="2").ask()}
            config['Grid_escalation_rate'] = [float(grid_esc_rate['rate'])] * 25
        else:
            grid_esc_array = {'rates': questionary.text("25 yearly escalation rates (%, comma-separated)", default="2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2").ask()}
            config['Grid_escalation_rate'] = [float(x.strip()) for x in grid_esc_array['rates'].split(',')]

        config['Grid_escalation'] = [x / 100 for x in config['Grid_escalation_rate']]

        # Monthly service charge
        print("\n→ Monthly Service Charge Configuration:")
        service_charge = {'system': questionary.select("Monthly service charge system", choices=[
            questionary.Choice('1 - Flat', value=1),
            questionary.Choice('2 - Tiered', value=2),
        ]).ask()}

        config['Monthly_fixed_charge_system'] = service_charge['system']

        if service_charge['system'] == 1:
            sc_flat = {'charge': questionary.text("Flat monthly charge ($)", default="27.16").ask()}
            config['SC_flat'] = float(sc_flat['charge'])

        elif service_charge['system'] == 2:
            sc_tiered = {}
            sc_tiered['tier1'] = questionary.text("Tier 1 charge ($)", default="34.29").ask()
            sc_tiered['limit1'] = questionary.text("Tier 1 limit (kWh)", default="800").ask()
            sc_tiered['tier2'] = questionary.text("Tier 2 charge ($)", default="46.54").ask()
            sc_tiered['limit2'] = questionary.text("Tier 2 limit (kWh)", default="1500").ask()
            sc_tiered['tier3'] = questionary.text("Tier 3 charge ($)", default="66.29").ask()
            sc_tiered['limit3'] = questionary.text("Tier 3 limit (kWh)", default="1500").ask()
            sc_tiered['tier4'] = questionary.text("Tier 4 charge ($)", default="66.29").ask()
            config['SC_1'] = float(sc_tiered['tier1'])
            config['Limit_SC_1'] = float(sc_tiered['limit1'])
            config['SC_2'] = float(sc_tiered['tier2'])
            config['Limit_SC_2'] = float(sc_tiered['limit2'])
            config['SC_3'] = float(sc_tiered['tier3'])
            config['Limit_SC_3'] = float(sc_tiered['limit3'])
            config['SC_4'] = float(sc_tiered['tier4'])


        # Electricity buying rate structure
        print("\n→ Electricity Buying Rate Structure:")
        print("""
Rate Structure Options:
1 = Flat rate
2 = Seasonal rate
3 = Monthly rate
4 = Tiered rate
5 = Seasonal tiered rate
6 = Monthly tiered rate
7 = Time of Use (TOU)
8 = Ultra Low Time of Use (ULO)
""")

        rate_struct = {'structure': questionary.select("Select electricity buying rate structure", choices=[
            questionary.Choice('1 - Flat rate', value=1),
            questionary.Choice('2 - Seasonal rate', value=2),
            questionary.Choice('3 - Monthly rate', value=3),
            questionary.Choice('4 - Tiered rate', value=4),
            questionary.Choice('5 - Seasonal tiered rate', value=5),
            questionary.Choice('6 - Monthly tiered rate', value=6),
            questionary.Choice('7 - Time of Use (TOU)', value=7),
            questionary.Choice('8 - Ultra Low TOU', value=8),
        ]).ask()}

        config['rateStructure'] = rate_struct['structure']

        # Handle each rate structure type with CORRECT numpy array structure
        if config['rateStructure'] == 1:  # Flat rate
            flat = {'price': questionary.text("Flat electricity price ($/kWh)", default="0.12").ask()}
            config['flatPrice'] = float(flat['price'])

        elif config['rateStructure'] == 2:  # Seasonal rate
            seasonal = {}
            seasonal['summer'] = questionary.text("Summer price ($/kWh)", default="0.0719").ask()
            seasonal['winter'] = questionary.text("Winter price ($/kWh)", default="0.0540").ask()
            seasonal['season'] = questionary.text("12 season indicators (0=winter, 1=summer, comma-separated)", default="0,0,0,0,0,1,1,1,1,0,0,0").ask()
            # CORRECT: Store as list for YAML, will be converted to np.array in backend
            config['seasonalPrices'] = [float(seasonal['summer']), float(seasonal['winter'])]
            config['season'] = [int(x.strip()) for x in seasonal['season'].split(',')]

        elif config['rateStructure'] == 3:  # Monthly rate
            monthly = {'prices': questionary.text("12 monthly prices ($/kWh, comma-separated)", default="0.082,0.088,0.088,0.095,0.095,0.113,0.129,0.129,0.113,0.095,0.095,0.082").ask()}
            config['monthlyPrices'] = [float(x.strip()) for x in monthly['prices'].split(',')]

        elif config['rateStructure'] == 4:  # Tiered rate
            tiered = {}
            tiered['tier1'] = questionary.text("Tier 1 price ($/kWh)", default="0.1018").ask()
            tiered['tier2'] = questionary.text("Tier 2 price ($/kWh)", default="0.1175").ask()
            tiered['tier3'] = questionary.text("Tier 3 price ($/kWh)", default="0.1175").ask()
            tiered['max1'] = questionary.text("Tier 1 max (kWh)", default="300").ask()
            tiered['max2'] = questionary.text("Tier 2 max (kWh)", default="999999").ask()
            tiered['max3'] = questionary.text("Tier 3 max (kWh)", default="999999").ask()
            config['tieredPrices'] = [float(tiered['tier1']), float(tiered['tier2']), float(tiered['tier3'])]
            config['tierMax'] = [float(tiered['max1']), float(tiered['max2']), float(tiered['max3'])]

        elif config['rateStructure'] == 5:  # Seasonal tiered rate
            seas_tiered = {}
            seas_tiered['summer_t1'] = questionary.text("Summer Tier 1 price ($/kWh)", default="0.075").ask()
            seas_tiered['summer_t2'] = questionary.text("Summer Tier 2 price ($/kWh)", default="0.091").ask()
            seas_tiered['summer_t3'] = questionary.text("Summer Tier 3 price ($/kWh)", default="0.091").ask()
            seas_tiered['winter_t1'] = questionary.text("Winter Tier 1 price ($/kWh)", default="0.075").ask()
            seas_tiered['winter_t2'] = questionary.text("Winter Tier 2 price ($/kWh)", default="0.091").ask()
            seas_tiered['winter_t3'] = questionary.text("Winter Tier 3 price ($/kWh)", default="0.091").ask()
            seas_tiered['summer_max1'] = questionary.text("Summer Tier 1 max (kWh)", default="600").ask()
            seas_tiered['summer_max2'] = questionary.text("Summer Tier 2 max (kWh)", default="999999").ask()
            seas_tiered['summer_max3'] = questionary.text("Summer Tier 3 max (kWh)", default="999999").ask()
            seas_tiered['winter_max1'] = questionary.text("Winter Tier 1 max (kWh)", default="1000").ask()
            seas_tiered['winter_max2'] = questionary.text("Winter Tier 2 max (kWh)", default="999999").ask()
            seas_tiered['winter_max3'] = questionary.text("Winter Tier 3 max (kWh)", default="999999").ask()
            seas_tiered['season'] = questionary.text("12 season indicators (0=winter, 1=summer)", default="0,0,0,0,1,1,1,1,1,1,0,0").ask()
            # CORRECT: 2D array structure for seasonal tiered
            config['seasonalTieredPrices'] = [
                [float(seas_tiered['summer_t1']), float(seas_tiered['summer_t2']), float(seas_tiered['summer_t3'])],
                [float(seas_tiered['winter_t1']), float(seas_tiered['winter_t2']), float(seas_tiered['winter_t3'])]
            ]
            config['seasonalTierMax'] = [
                [float(seas_tiered['summer_max1']), float(seas_tiered['summer_max2']),
                 float(seas_tiered['summer_max3'])],
                [float(seas_tiered['winter_max1']), float(seas_tiered['winter_max2']),
                 float(seas_tiered['winter_max3'])]
            ]
            config['season'] = [int(x.strip()) for x in seas_tiered['season'].split(',')]

        elif config['rateStructure'] == 6:  # Monthly tiered rate
            print("→ Monthly Tiered Rate: Enter prices and limits for all 12 months")
            monthly_tiered_prices = []
            monthly_tier_limits = []

            for month in range(1, 13):
                month_data = {}
                month_data['t1'] = questionary.text(f"Month {month} - Tier 1 price ($/kWh)", default="0.0874").ask()
                month_data['t2'] = questionary.text(f"Month {month} - Tier 2 price ($/kWh)", default="0.1027").ask()
                month_data['t3'] = questionary.text(f"Month {month} - Tier 3 price ($/kWh)", default="0.1027").ask()
                month_data['max1'] = questionary.text(f"Month {month} - Tier 1 max (kWh)", default="600").ask()
                month_data['max2'] = questionary.text(f"Month {month} - Tier 2 max (kWh)", default="999999").ask()
                month_data['max3'] = questionary.text(f"Month {month} - Tier 3 max (kWh)", default="999999").ask()
                monthly_tiered_prices.append(
                    [float(month_data['t1']), float(month_data['t2']), float(month_data['t3'])])
                monthly_tier_limits.append(
                    [float(month_data['max1']), float(month_data['max2']), float(month_data['max3'])])

            config['monthlyTieredPrices'] = monthly_tiered_prices
            config['monthlyTierLimits'] = monthly_tier_limits

        elif config['rateStructure'] == 7:  # Time of Use (TOU)
            print("\n→ Time of Use Rate Configuration:")
            tou = {}
            tou['on_summer'] = questionary.text("On-peak price - Summer ($/kWh)", default="0.151").ask()
            tou['on_winter'] = questionary.text("On-peak price - Winter ($/kWh)", default="0.113").ask()
            tou['mid_summer'] = questionary.text("Mid-peak price - Summer ($/kWh)", default="0.102").ask()
            tou['mid_winter'] = questionary.text("Mid-peak price - Winter ($/kWh)", default="0.094").ask()
            tou['off_summer'] = questionary.text("Off-peak price - Summer ($/kWh)", default="0.074").ask()
            tou['off_winter'] = questionary.text("Off-peak price - Winter ($/kWh)", default="0.074").ask()
            tou['on_hours_summer'] = questionary.text("On-peak hours - Summer (comma-separated, e.g., 11,12,13,14,15,16)", default="11,12,13,14,15,16").ask()
            tou['on_hours_winter'] = questionary.text("On-peak hours - Winter (comma-separated)", default="17,18,19,20").ask()
            tou['mid_hours_summer'] = questionary.text("Mid-peak hours - Summer (comma-separated)", default="7,8,9,10,17,18,19,20,21").ask()
            tou['mid_hours_winter'] = questionary.text("Mid-peak hours - Winter (comma-separated)", default="7,8,9,10,11,12,13,14,15,16,21").ask()
            tou['season'] = questionary.text("12 season indicators (0=winter, 1=summer)", default="0,0,0,0,0,1,1,1,1,0,0,0").ask()
            tou['treat_special_days_as_offpeak'] = questionary.confirm("Treat weekends/holidays as off-peak?", default=True).ask()

            # CORRECT structure: numpy arrays with [summer, winter] format
            config['onPrice'] = [float(tou['on_summer']), float(tou['on_winter'])]
            config['midPrice'] = [float(tou['mid_summer']), float(tou['mid_winter'])]
            config['offPrice'] = [float(tou['off_summer']), float(tou['off_winter'])]

            # CORRECT: Store hours as nested lists [summer_hours, winter_hours]
            on_hours_summer = [int(x.strip()) for x in tou['on_hours_summer'].split(',')]
            on_hours_winter = [int(x.strip()) for x in tou['on_hours_winter'].split(',')]
            mid_hours_summer = [int(x.strip()) for x in tou['mid_hours_summer'].split(',')]
            mid_hours_winter = [int(x.strip()) for x in tou['mid_hours_winter'].split(',')]

            config['onHours'] = [on_hours_summer, on_hours_winter]
            config['midHours'] = [mid_hours_summer, mid_hours_winter]
            config['season'] = [int(x.strip()) for x in tou['season'].split(',')]
            config['treat_special_days_as_offpeak'] = tou['treat_special_days_as_offpeak']

        elif config['rateStructure'] == 8:  # Ultra Low TOU
            print("\n→ Ultra Low Time of Use Rate Configuration:")
            ulo = {}
            ulo['on_summer'] = questionary.text("On-peak price - Summer ($/kWh)", default="0.284").ask()
            ulo['on_winter'] = questionary.text("On-peak price - Winter ($/kWh)", default="0.284").ask()
            ulo['mid_summer'] = questionary.text("Mid-peak price - Summer ($/kWh)", default="0.122").ask()
            ulo['mid_winter'] = questionary.text("Mid-peak price - Winter ($/kWh)", default="0.122").ask()
            ulo['off_summer'] = questionary.text("Off-peak price - Summer ($/kWh)", default="0.076").ask()
            ulo['off_winter'] = questionary.text("Off-peak price - Winter ($/kWh)", default="0.076").ask()
            ulo['ultra_summer'] = questionary.text("Ultra-low price - Summer ($/kWh)", default="0.028").ask()
            ulo['ultra_winter'] = questionary.text("Ultra-low price - Winter ($/kWh)", default="0.028").ask()
            ulo['on_hours_summer'] = questionary.text("On-peak hours - Summer (comma-separated)", default="16,17,18,19,20").ask()
            ulo['on_hours_winter'] = questionary.text("On-peak hours - Winter (comma-separated)", default="16,17,18,19,20").ask()
            ulo['mid_hours_summer'] = questionary.text("Mid-peak hours - Summer (comma-separated)", default="7,8,9,10,11,12,13,14,15,21,22").ask()
            ulo['mid_hours_winter'] = questionary.text("Mid-peak hours - Winter (comma-separated)", default="7,8,9,10,11,12,13,14,15,21,22").ask()
            ulo['ultra_hours_summer'] = questionary.text("Ultra-low hours - Summer (comma-separated)", default="23,0,1,2,3,4,5,6").ask()
            ulo['ultra_hours_winter'] = questionary.text("Ultra-low hours - Winter (comma-separated)", default="23,0,1,2,3,4,5,6").ask()
            ulo['season'] = questionary.text("12 season indicators (0=winter, 1=summer)", default="0,0,0,0,0,0,0,0,0,0,0,0").ask()
            ulo['treat_special_days_as_offpeak'] = questionary.confirm("Treat weekends/holidays as off-peak?", default=True).ask()

            # CORRECT structure for ULO
            config['onPrice'] = [float(ulo['on_summer']), float(ulo['on_winter'])]
            config['midPrice'] = [float(ulo['mid_summer']), float(ulo['mid_winter'])]
            config['offPrice'] = [float(ulo['off_summer']), float(ulo['off_winter'])]
            config['ultraLowPrice'] = [float(ulo['ultra_summer']), float(ulo['ultra_winter'])]

            on_hours_summer = [int(x.strip()) for x in ulo['on_hours_summer'].split(',')]
            on_hours_winter = [int(x.strip()) for x in ulo['on_hours_winter'].split(',')]
            mid_hours_summer = [int(x.strip()) for x in ulo['mid_hours_summer'].split(',')]
            mid_hours_winter = [int(x.strip()) for x in ulo['mid_hours_winter'].split(',')]
            ultra_hours_summer = [int(x.strip()) for x in ulo['ultra_hours_summer'].split(',')]
            ultra_hours_winter = [int(x.strip()) for x in ulo['ultra_hours_winter'].split(',')]

            config['onHours'] = [on_hours_summer, on_hours_winter]
            config['midHours'] = [mid_hours_summer, mid_hours_winter]
            config['ultraLowHours'] = [ultra_hours_summer, ultra_hours_winter]
            config['season'] = [int(x.strip()) for x in ulo['season'].split(',')]
            config['treat_special_days_as_offpeak'] = ulo['treat_special_days_as_offpeak']

        # Electricity selling rate structure
        print("\n→ Electricity Selling Rate Structure:")
        sell_struct = {'structure': questionary.select("Select electricity selling rate structure", choices=[
            questionary.Choice('1 - Flat rate', value=1),
            questionary.Choice('2 - Monthly rate', value=2),
            questionary.Choice('3 - Same as buying rate', value=3),
        ]).ask()}

        config['sellStructure'] = sell_struct['structure']

        if config['sellStructure'] == 1:
            sell_flat = {'price': questionary.text("Flat selling price ($/kWh)", default="0.1").ask()}
            config['Csell_flat'] = float(sell_flat['price'])

        elif config['sellStructure'] == 2:
            sell_monthly = {'prices': questionary.text("12 monthly selling prices ($/kWh, comma-separated)", default="0.05799,0.04829,0.04621,0.04256,0.04030,0.03991,0.03963,0.03976,0.03781,0.03656,0.03615,0.03461").ask()}
            config['monthlysellprices'] = [float(x.strip()) for x in sell_monthly['prices'].split(',')]

        # Grid emissions and constraints
        grid_emissions = {}
        grid_emissions['e_co2'] = questionary.text("Grid CO2 emissions (kg/kWh)", default="0").ask()
        grid_emissions['e_so2'] = questionary.text("Grid SO2 emissions (kg/kWh)", default="0").ask()
        grid_emissions['e_nox'] = questionary.text("Grid NOx emissions (kg/kWh)", default="0").ask()
        grid_emissions['pbuy_max'] = questionary.text("Maximum grid purchase (kW)", default="50").ask()
        grid_emissions['psell_max'] = questionary.text("Maximum grid sell (kW)", default="50").ask()

        config['E_CO2'] = float(grid_emissions['e_co2'])
        config['E_SO2'] = float(grid_emissions['e_so2'])
        config['E_NOx'] = float(grid_emissions['e_nox'])
        config['Pbuy_max'] = float(grid_emissions['pbuy_max'])
        config['Psell_max'] = float(grid_emissions['psell_max'])

    # ============================================================================
    # SECTION 18: NATURAL GAS CONFIGURATION (ONLY IF HP SELECTED)
    # ============================================================================
    if config['HP'] == 1:
        print("\n" + "=" * 80)
        print("SECTION 18: NATURAL GAS CONFIGURATION (Required for Heat Pump)")
        print("=" * 80)

        ng_check = {'enable_ng': questionary.confirm("Configure Natural Gas rates? (needed for HP)", default=True).ask()}

        if ng_check['enable_ng']:

            # ── NG physical constants ─────────────────────────────────────────
            ng_params = {}
            ng_params['energy_content'] = questionary.text("NG energy content (kWh/m³)", default="10.81").ask()
            ng_params['furnace_eff'] = questionary.text("Furnace efficiency", default="0.94").ask()
            config['NG_energycontent'] = float(ng_params['energy_content'])
            config['Furnace_eff']      = float(ng_params['furnace_eff'])

            # ── Unit system: all prices entered in this unit ──────────────────
            ng_unit_ans = {'unit': questionary.select("Your NG utility bills prices in:", choices=[
                questionary.Choice('$/m³  (cubic metres — most of Canada/Europe)', value='m3'),
                questionary.Choice('$/kWh (already in energy units)', value='kwh'),
                questionary.Choice('$/therm (USA — 1 therm = 29.3001 kWh)', value='therms'),
            ]).ask()}
            config['NG_unit'] = ng_unit_ans['unit']
            u = config['NG_unit']           # shorthand for labels below
            ulabel = {'m3': '$/m³', 'kwh': '$/kWh', 'therms': '$/therm'}[u]
            # Volume unit for tier limits
            vollabel = {'m3': 'm³', 'kwh': 'kWh', 'therms': 'therms'}[u]

            # ── NG rate structure ─────────────────────────────────────────────
            ng_rate = {'structure': questionary.select("NG rate structure", choices=[
                questionary.Choice('1 - Flat rate', value=1),
                questionary.Choice('2 - Seasonal rate', value=2),
                questionary.Choice('3 - Monthly rate', value=3),
                questionary.Choice('4 - Tiered rate', value=4),
                questionary.Choice('5 - Seasonal tiered rate', value=5),
                questionary.Choice('6 - Monthly tiered rate', value=6),
                questionary.Choice('7 - Therms-based tiered (e.g. PG&E G-1)', value=7),
                questionary.Choice('8 - 4-tier m³-based (e.g. Enbridge EGD)', value=8),
            ]).ask()}
            config['rateStructure_NG'] = ng_rate['structure']

            if ng_rate['structure'] == 1:  # Flat
                ans = {'price': questionary.text(f"Flat NG price ({ulabel})", default="0.28").ask()}
                config['flatPrice_NG'] = float(ans['price'])

            elif ng_rate['structure'] == 2:  # Seasonal
                ans = {}
                ans['summer'] = questionary.text(f"Summer price ({ulabel})", default="0.0719").ask()
                ans['winter'] = questionary.text(f"Winter price ({ulabel})", default="0.0540").ask()
                ans['season'] = questionary.text("12 season flags (0=winter,1=summer)", default="0,0,0,0,0,1,1,1,1,0,0,0").ask()
                config['seasonalPrices_NG'] = [float(ans['summer']), float(ans['winter'])]
                config['season_NG']         = [int(x.strip()) for x in ans['season'].split(',')]

            elif ng_rate['structure'] == 3:  # Monthly
                ans = {'prices': questionary.text(f"12 monthly prices ({ulabel}, comma-sep)", default="0.54207,0.53713,0.38689,0.30496,0.28689,0.28168,0.30205,0.28956,0.26501,0.26492,0.3108,0.40715").ask()}
                config['monthlyPrices_NG'] = [float(x.strip()) for x in ans['prices'].split(',')]

            elif ng_rate['structure'] == 4:  # Tiered
                ans = {}
                ans['t1'] = questionary.text(f"Tier 1 price ({ulabel})", default="0.1018").ask()
                ans['t2'] = questionary.text(f"Tier 2 price ({ulabel})", default="0.1175").ask()
                ans['t3'] = questionary.text(f"Tier 3 price ({ulabel})", default="0.1175").ask()
                ans['max1'] = questionary.text(f"Tier 1 max ({vollabel}/month)", default="300").ask()
                ans['max2'] = questionary.text(f"Tier 2 max ({vollabel}/month)", default="999999").ask()
                ans['max3'] = questionary.text(f"Tier 3 max ({vollabel}/month)", default="999999").ask()
                config['tieredPrices_NG'] = [float(ans['t1']), float(ans['t2']), float(ans['t3'])]
                config['tierMax_NG']      = [float(ans['max1']), float(ans['max2']), float(ans['max3'])]

            elif ng_rate['structure'] == 5:  # Seasonal tiered
                ans = {}
                ans['su_t1'] = questionary.text(f"Summer Tier 1 ({ulabel})", default="0.075").ask()
                ans['su_t2'] = questionary.text(f"Summer Tier 2 ({ulabel})", default="0.091").ask()
                ans['su_t3'] = questionary.text(f"Summer Tier 3 ({ulabel})", default="0.091").ask()
                ans['wi_t1'] = questionary.text(f"Winter Tier 1 ({ulabel})", default="0.075").ask()
                ans['wi_t2'] = questionary.text(f"Winter Tier 2 ({ulabel})", default="0.091").ask()
                ans['wi_t3'] = questionary.text(f"Winter Tier 3 ({ulabel})", default="0.091").ask()
                ans['su_max1'] = questionary.text(f"Summer Tier 1 max ({vollabel}/month)", default="600").ask()
                ans['su_max2'] = questionary.text(f"Summer Tier 2 max ({vollabel}/month)", default="999999").ask()
                ans['su_max3'] = questionary.text(f"Summer Tier 3 max ({vollabel}/month)", default="999999").ask()
                ans['wi_max1'] = questionary.text(f"Winter Tier 1 max ({vollabel}/month)", default="1000").ask()
                ans['wi_max2'] = questionary.text(f"Winter Tier 2 max ({vollabel}/month)", default="999999").ask()
                ans['wi_max3'] = questionary.text(f"Winter Tier 3 max ({vollabel}/month)", default="999999").ask()
                ans['season'] = questionary.text("12 season flags (0=winter,1=summer)", default="0,0,0,0,1,1,1,1,1,1,0,0").ask()
                config['seasonalTieredPrices_NG'] = [
                    [float(ans['su_t1']), float(ans['su_t2']), float(ans['su_t3'])],
                    [float(ans['wi_t1']), float(ans['wi_t2']), float(ans['wi_t3'])],
                ]
                config['seasonalTierMax_NG'] = [
                    [float(ans['su_max1']), float(ans['su_max2']), float(ans['su_max3'])],
                    [float(ans['wi_max1']), float(ans['wi_max2']), float(ans['wi_max3'])],
                ]
                config['season_NG'] = [int(x.strip()) for x in ans['season'].split(',')]

            elif ng_rate['structure'] == 6:  # Monthly tiered
                print(f"→ Monthly Tiered NG Rate: enter prices ({ulabel}) and limits ({vollabel}/month) for all 12 months")
                monthly_tiered_ng = []
                monthly_limits_ng = []
                for month in range(1, 13):
                    m = {}
                    m['t1'] = questionary.text(f"Month {month} Tier 1 ({ulabel})", default="0.404").ask()
                    m['t2'] = questionary.text(f"Month {month} Tier 2 ({ulabel})", default="0.509").ask()
                    m['t3'] = questionary.text(f"Month {month} Tier 3 ({ulabel})", default="0.509").ask()
                    m['max1'] = questionary.text(f"Month {month} Tier 1 max ({vollabel})", default="343").ask()
                    m['max2'] = questionary.text(f"Month {month} Tier 2 max ({vollabel})", default="999999").ask()
                    m['max3'] = questionary.text(f"Month {month} Tier 3 max ({vollabel})", default="999999").ask()
                    monthly_tiered_ng.append([float(m['t1']),   float(m['t2']),   float(m['t3'])])
                    monthly_limits_ng.append( [float(m['max1']), float(m['max2']), float(m['max3'])])
                config['monthlyTieredPrices_NG'] = monthly_tiered_ng
                config['monthlyTierLimits_NG']   = monthly_limits_ng

            elif ng_rate['structure'] == 7:  # PG&E G-1 style (therms-based 2-tier)
                print("→ Therms-based tiered rate (e.g. PG&E G-1 Territory T)")
                print("  Prices must be in $/therm. Tier limits are monthly baseline therms per month.")
                ans = {}
                ans['base_rate'] = questionary.text("Baseline tier price ($/therm)", default="2.41665").ask()
                ans['excess_rate'] = questionary.text("Excess tier price ($/therm)", default="2.91799").ask()
                ans['baseline_therms'] = questionary.text("12 monthly baseline therms (comma-sep)", default="52.08,36.68,40.61,16.80,17.36,16.80,17.36,17.36,16.80,17.36,39.30,52.08").ask()
                config['ng7_base_rate']       = float(ans['base_rate'])
                config['ng7_excess_rate']     = float(ans['excess_rate'])
                config['ng7_baseline_therms'] = [float(x.strip()) for x in ans['baseline_therms'].split(',')]

            elif ng_rate['structure'] == 8:  # Enbridge EGD style (m³ 4-tier)
                print("→ 4-tier m³-based rate (e.g. Enbridge EGD, Toronto)")
                print("  Prices in $/m³. Tier limits are cumulative monthly m³ thresholds.")
                ans = {}
                ans['t1'] = questionary.text("Tier 1 price ($/m³) — first 30 m³", default="0.10638").ask()
                ans['t2'] = questionary.text("Tier 2 price ($/m³) — next 55 m³", default="0.10026").ask()
                ans['t3'] = questionary.text("Tier 3 price ($/m³) — next 85 m³", default="0.09547").ask()
                ans['t4'] = questionary.text("Tier 4 price ($/m³) — over 170 m³", default="0.09190").ask()
                ans['lim1'] = questionary.text("Tier 1 upper limit (m³/month)", default="30").ask()
                ans['lim2'] = questionary.text("Tier 2 upper limit (m³/month)", default="85").ask()
                ans['lim3'] = questionary.text("Tier 3 upper limit (m³/month)", default="170").ask()
                config['ng8_prices'] = [float(ans['t1']), float(ans['t2']), float(ans['t3']), float(ans['t4'])]
                config['ng8_limits'] = [float(ans['lim1']), float(ans['lim2']), float(ans['lim3'])]

            # ── NG economics (all monetary values in chosen unit) ─────────────
            ng_econ = {}
            ng_econ['annual_exp'] = questionary.text("Annual NG expenses ($)", default="0").ask()
            ng_econ['sale_tax'] = questionary.text("NG sale tax rate (%)", default="13").ask()
            ng_econ['grid_tax'] = questionary.text(f"NG grid tax ({ulabel})", default="0.11").ask()
            ng_econ['credit'] = questionary.text(f"NG grid credit ({ulabel})", default="0").ask()
            config['Annual_expenses_NG']    = float(ng_econ['annual_exp'])
            config['Grid_sale_tax_rate_NG'] = float(ng_econ['sale_tax'])
            config['Grid_Tax_NG']           = config['Grid_sale_tax_rate_NG'] / 100
            config['Grid_Tax_amount_NG']    = float(ng_econ['grid_tax'])   # raw; converted in config_loader
            config['Grid_credit_NG']        = float(ng_econ['credit'])     # per-unit; converted to $/kWh in config_loader

            # ── NG escalation ─────────────────────────────────────────────────
            ng_esc = {'projection': questionary.select("NG escalation projection", choices=[
                questionary.Choice('1 - Flat yearly rate', value=1),
                questionary.Choice('2 - 25 different yearly rates', value=2),
            ]).ask()}
            config['Grid_escalation_projection_NG'] = ng_esc['projection']
            if ng_esc['projection'] == 1:
                flat_r = {'rate': questionary.text("NG yearly escalation rate (%)", default="2").ask()}
                config['Grid_escalation_rate_NG'] = [float(flat_r['rate'])] * 25
            else:
                arr_r = {'rates': questionary.text("25 yearly NG escalation rates (%, comma-sep)").ask()}
                config['Grid_escalation_rate_NG'] = [float(x.strip()) for x in arr_r['rates'].split(',')]
            config['Grid_escalation_NG'] = [x / 100 for x in config['Grid_escalation_rate_NG']]

            # ── NG service charge ─────────────────────────────────────────────
            ng_sc = {'charge': questionary.text("Flat NG monthly service charge ($)", default="18.59").ask()}
            config['Monthly_fixed_charge_system_NG'] = 1
            config['SC_flat_NG']       = float(ng_sc['charge'])
            config['Service_charge_NG'] = config['SC_flat_NG']

            print("\n✅ Natural Gas configuration complete!")
        else:
            print("\n⚠️  Skipping NG configuration - HP will use default settings")
    else:
        # HP not selected, NG_Grid stays 0
        print("\n→ Heat Pump not selected - Natural Gas configuration skipped")

    # ============================================================================
    # SAVE FINAL CONFIGURATION
    # ============================================================================
    print("\n" + "=" * 80)
    print("SAVING FINAL CONFIGURATION")
    print("=" * 80)

    output_path = input_dir / 'samapy_config_COMPLETE.yaml'

    with open(output_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)

    print(f"\n✅ Configuration saved → {output_path}")
    print(f"\n   Run optimization:  samapy-run\n")

    return config


if __name__ == '__main__':
    try:
        config = main()
    except KeyboardInterrupt:
        print("\n\n❌ Configuration cancelled by user.")
    except Exception as e:
        print(f"\n\n❌ Error: {e}")
        import traceback

        traceback.print_exc()
        raise