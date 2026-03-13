# SAMA
## Solar Alone Multi-Objective Advisor

**Comprehensive User Documentation | v4**

*Free Appropriate Sustainability Technology (FAST) Research Group | Western University, London, Ontario, Canada*

---

## Table of Contents

- [0. Before You Start: Installing Python](#0-before-you-start-installing-python)
  - [0.1 Checking if Python is Already Installed](#01-checking-if-python-is-already-installed)
  - [0.2 Installing Python on Windows](#02-installing-python-on-windows)
  - [0.3 Installing Python on macOS](#03-installing-python-on-macos)
  - [0.4 Opening a Terminal](#04-opening-a-terminal)
  - [0.5 Verifying pip](#05-verifying-pip)
- [1. Overview](#1-overview)
  - [1.1 What SAMA Optimizes](#11-what-sama-optimizes)
  - [1.2 Key Capabilities](#12-key-capabilities)
  - [1.3 Comparison to HOMER Pro](#13-comparison-to-homer-pro)
- [2. Three Ways to Use SAMA](#2-three-ways-to-use-sama)
- [3. Method 1: Using the Raw Python Code](#3-method-1--using-the-raw-python-code)
  - [3.1 Prerequisites](#31-prerequisites)
  - [3.2 Downloading and Opening the Code](#32-downloading-and-opening-the-code)
  - [3.3 Configuring Parameters Directly in Input_Data.py](#33-configuring-parameters-directly-in-input_datapy)
  - [3.4 Running the Optimization](#34-running-the-optimization)
- [4. Method 2: Using the Python Package (pip install samapy)](#4-method-2--using-the-python-package-pip-install-samapy)
  - [4.1 Installation](#41-installation)
  - [4.2 Step 1: Run the Configuration Wizard](#42-step-1--run-the-configuration-wizard)
  - [4.3 Step 2: Run the Optimization](#43-step-2--run-the-optimization)
  - [4.4 Step 3: View Results](#44-step-3--view-results)
  - [4.5 Using the Python API](#45-using-the-python-api)
  - [4.6 Using Your Own Custom Data Files](#46-using-your-own-custom-data-files)
  - [4.7 Typical Workflow for a New Project Location](#47-typical-workflow-for-a-new-project-location)
  - [4.8 Understanding the Working Directory](#48-understanding-the-working-directory)
  - [4.9 What to Expect on the First Run](#49-what-to-expect-on-the-first-run)
- [5. Method 3: Windows .exe Application (Alpha)](#5-method-3--windows-exe-application-alpha)
- [6. The Configuration File (sama_config_COMPLETE.yaml)](#6-the-configuration-file-sama_config_completeyaml)
  - [6.1 File Structure Overview](#61-file-structure-overview)
  - [6.2 Key Parameter Reference](#62-key-parameter-reference)
- [7. Optimization Algorithms](#7-optimization-algorithms)
  - [7.1 Particle Swarm Optimization (PSO)](#71-particle-swarm-optimization-pso)
  - [7.2 Advanced Differential Evolution (ADE)](#72-advanced-differential-evolution-ade)
  - [7.3 Improved Artificial Bee Colony (ABC)](#73-improved-artificial-bee-colony-abc)
  - [7.4 Grey Wolf Optimizer (GWO)](#74-grey-wolf-optimizer-gwo)
  - [7.5 Choosing an Algorithm](#75-choosing-an-algorithm)
- [8. Energy Management System (EMS)](#8-energy-management-system-ems)
  - [8.1 Base Dispatch (EMS.py)](#81-base-dispatch-emspy)
  - [8.2 EV Smart Charging and V2X (EMS_EV.py)](#82-ev-smart-charging-and-v2x-ems_evpy)
  - [8.3 Heat Pump Energy Attribution (EMS_HP.py)](#83-heat-pump-energy-attribution-ems_hppy)
- [9. Component Models](#9-component-models)
  - [9.1 Solar PV](#91-solar-pv)
  - [9.2 Wind Turbine](#92-wind-turbine)
  - [9.3 Battery Storage](#93-battery-storage)
  - [9.4 Diesel Generator](#94-diesel-generator)
  - [9.5 Heat Pump Models](#95-heat-pump-models)
- [10. Financial Model](#10-financial-model)
  - [10.1 Net Present Cost (NPC)](#101-net-present-cost-npc)
  - [10.2 Levelized Cost of Energy (LCOE)](#102-levelized-cost-of-energy-lcoe)
  - [10.3 Levelized Emissions (LEM)](#103-levelized-emissions-lem)
  - [10.4 25-Year Cash Flow Projection](#104-25-year-cash-flow-projection)
- [11. Bundled Data Files (content/ folder)](#11-bundled-data-files-content-folder)
- [12. Input Data Formats](#12-input-data-formats)
  - [12.1 Electrical Load CSV (Eload.csv)](#121-electrical-load-csv-eloadcsv)
  - [12.2 METEO.csv (SAM NSRDB format)](#122-meteocsy-sam-nsrdb-format)
  - [12.3 Plane-of-Array Irradiance CSV (Irradiance.csv)](#123-plane-of-array-irradiance-csv-irradiancecsv)
  - [12.4 house_load.xlsx (Thermal Load)](#124-house_loadxlsx-thermal-load)
- [13. Troubleshooting](#13-troubleshooting)
- [14. Quick Reference: Minimum Working Example](#14-quick-reference--minimum-working-example)
- [15. Package Module Structure](#15-package-module-structure)
- [16. License and Citation](#16-license-and-citation)
  - [16.1 License](#161-license)
  - [16.2 Authors](#162-authors)
  - [16.3 Citing SAMA](#163-citing-samapy)
- [17. Examples](#17-examples)
- [18. Sample Optimization Output](#18-sample-optimization-output)
- [19. Technical Framework and Mathematical Formulation](#19-technical-framework-and-mathematical-formulation)
- [20. References](#20-references)
- [21. SAMA Publications and Further Reading](#21-sama-publications-and-further-reading)
- [Appendix A: Where to Find Input Data for SAMA](#appendix-a-where-to-find-input-data-for-sama)

---

## 0. Before You Start: Installing Python

SAMA requires **Python 3.9 or higher**. If you do not already have Python installed on your computer, follow the steps below before proceeding to any other section of this guide.

### 0.1 Checking if Python is Already Installed

Open a terminal (Command Prompt on Windows, Terminal on macOS/Linux) and type:

```bash
python --version
```

If you see a version number such as `Python 3.11.4`, Python is already installed and you can skip this section. If you see an error or a version below 3.9, follow the steps below.

### 0.2 Installing Python on Windows

1. Go to https://www.python.org/downloads/ and click **Download Python** (choose the latest 3.x version).
2. Run the installer. On the first screen, **check the box that says "Add Python to PATH"** before clicking Install Now. This is the most important step, without it, Python commands will not work in the terminal.
3. Once installation is complete, close and reopen your Command Prompt and verify with:

```bash
python --version
```

### 0.3 Installing Python on macOS

1. Go to https://www.python.org/downloads/ and download the macOS installer.
2. Run the `.pkg` installer and follow the on-screen steps.
3. Open Terminal and verify with:

```bash
python3 --version
```

### 0.4 Opening a Terminal

Throughout this guide, instructions refer to running commands in a terminal. Here is how to open one on each platform:

- **Windows:** Press the Windows key, type `cmd`, and press Enter. Or open File Explorer, navigate to your project folder, click the address bar, type `cmd`, and press Enter to open a terminal directly in that folder.
- **macOS:** Press `Command + Space`, type `Terminal`, and press Enter.
- **Linux:** Press `Ctrl + Alt + T`.

### 0.5 Verifying pip

`pip` is the Python package installer and is included automatically with Python 3.9 and later. Verify it is available:

```bash
pip --version
```

If pip is not found, run:

```bash
python -m ensurepip --upgrade
```

---

## 1. Overview

SAMA (Solar Alone Multi-Objective Advisor) is an open-source Python platform for the design, sizing, and operation optimization of hybrid renewable energy microgrids. It performs location-specific techno-economic assessment using site-dependent electrical and thermal load profiles, meteorological data, component degradation models, and utility-specific billing structures.

Unlike proprietary tools such as HOMER Pro and HOMER Grid, which are now closed-source and license-restricted, SAMA provides **transparent optimization algorithms**, **fully extensible architecture**, and **no cost barriers** for researchers, practitioners, and resource-constrained institutions.

### 1.1 What SAMA Optimizes

SAMA finds the optimal combination of five design variables:

| Variable | Description |
|----------|-------------|
| `Npv` | Number of PV modules in units of rated PV capacity (kW) |
| `Nwt` | Number of wind turbines |
| `Nbat` | Number of battery packs |
| `N_DG` | Number (size) of diesel generator units |
| `Cn_I` | Inverter capacity (kW) |

The search space is bounded by `VarMin` and `VarMax` (default: 0 to 60 for each variable). The optimizer explores this space to minimize the primary objective, **Net Present Cost (NPC)**, while satisfying reliability, renewable fraction, budget, and rooftop-area constraints. When `EM` is set to `1`, emissions (Levelized Emissions, LEM) are included as a second objective.

### 1.2 Key Capabilities

| Capability | Detail |
|------------|--------|
| **System components** | Solar PV, wind turbines, battery storage (lead-acid or lithium-ion), diesel generator, utility grid, air-source heat pump (Bosch or Goodman), electric vehicle (V2X) |
| **Optimization algorithms** | Particle Swarm Optimization (PSO), Advanced Differential Evolution (ADE), Artificial Bee Colony (ABC), Grey Wolf Optimizer (GWO) |
| **Energy management** | 8,760-hour hourly dispatch with EMS, EV smart charging/discharging with V2X lookahead arbitrage, heat pump thermal-electric coupling |
| **Electricity tariffs** | 8 rate structures: flat, seasonal, monthly, tiered, seasonal tiered, monthly tiered, Time-of-Use (TOU), Ultra-Low TOU (ULO) |
| **Natural gas tariffs** | 8 rate structures for NG: flat, seasonal, monthly, tiered, seasonal tiered, monthly tiered, PG&E G-1 therms-based, Enbridge EGD 4-tier m³-based |
| **Financial analysis** | 25-year Net Present Cost, LCOE, levelized emissions, escalation rates, RE incentives (e.g. 30% federal tax credit), net metering reconciliation |
| **Load inputs** | 10 electrical load input modes (CSV, monthly/annual averages, generic profiles), thermal load from Excel |
| **Weather inputs** | SAM NSRDB API integration, user CSV files, monthly/annual averages |
| **Outputs** | Convergence plots, cash flow charts, energy distribution, battery SOC, grid interconnection, EV energy charts, HP performance charts, hourly data CSV |
| **License** | GNU General Public License v3.0 (GPL-3.0), free to use, modify, and distribute |

### 1.3 Comparison to HOMER Pro

SAMA has been validated against HOMER Pro in diverse climatic zones (Sacramento, California and New Bern, North Carolina) and achieves close agreement in optimal system configurations and performance metrics. SAMA additionally supports constraint scenarios such as rooftop area limits, net-metering caps, renewable fraction requirements, and loss-of-power-supply probability, which are handled more flexibly than in proprietary tools.

---

## 2. Three Ways to Use SAMA

SAMA can be used in three different ways depending on your background and needs:

| Method | Who It Is For | How to Get It |
|--------|--------------|---------------|
| **Method 1:** Raw Python Code | Researchers, developers, and advanced users who want to modify algorithms or parameters directly in the source code | Download from GitHub and open in a Python IDE such as PyCharm |
| **Method 2:** Python Package (pip) | Users who want to run SAMA as a command-line tool or use it in their own Python scripts without modifying internals | `pip install samapy` (available as the SAMA Py Package) |
| **Method 3:** Windows .exe Application (Alpha) | Non-technical users who want a point-and-click graphical user interface without installing Python. Note: this version is currently in the alpha development phase. | Available on request from the developers |

---

## 3. Method 1: Using the Raw Python Code

This method gives you full access to all source files. You can directly edit `Input_Data.py` to set your parameters and run the optimizer scripts directly. This is the approach used in the `examples/` folder.

### 3.1 Prerequisites

- Python 3.9 or higher
- A Python IDE such as PyCharm (recommended), VS Code, or Spyder
- Git (optional, for cloning the repository) or a browser to download the ZIP from GitHub

### 3.2 Downloading and Opening the Code

> **GitHub Repository (Method 1 source code):**
> https://github.com/Sas1997/SAMA/tree/main/Backend%20Codes/SAMA%20V2.0.1-GitHub

1. Open the URL above in your browser.
2. Click the green **Code** button and choose **Download ZIP**, or clone with Git:

```bash
git clone https://github.com/Sas1997/SAMA.git
```

3. Extract the ZIP (or navigate into the cloned folder) and open the folder `Backend Codes/SAMA V2.0.1-GitHub` in PyCharm (or your preferred IDE).
4. PyCharm will detect the project structure automatically. Set up a Python interpreter (Python 3.9 or higher) in **File > Settings > Project > Python Interpreter**.
5. Install the required packages. You can do this in the PyCharm terminal or any command prompt:

```bash
pip install numpy pandas scipy numba matplotlib openpyxl seaborn numpy-financial questionary PyYAML
```

6. Verify the setup by running:

```bash
python -c "from samapy.core.Input_Data import InData; print('OK')"
```

### 3.3 Configuring Parameters Directly in Input_Data.py

Open `samapy/core/Input_Data.py` in your IDE. All parameters are set in the `Input_Data.__init__` method. The key sections to edit are described below.

#### 3.3.1 System Components

Enable or disable each component by setting its flag to `1` (included) or `0` (excluded):

```python
self.PV = 1        # Solar PV
self.WT = 0        # Wind turbine
self.DG = 1        # Diesel generator
self.Bat = 1       # Battery storage
self.Grid = 1      # Utility grid connection
self.HP = 1        # Heat pump (Goodman or Bosch)
self.EV = 0        # Electric vehicle
self.Lead_acid = 0 # Lead-acid battery
self.Li_ion = 1    # Li-ion battery
```

#### 3.3.2 Optimization Search Space

`VarMin` and `VarMax` define the lower and upper bounds for the five design variables `[Npv, Nwt, Nbat, N_DG, Cn_I]`:

```python
self.VarMin = np.array([0, 0, 0, 0, 0])        # lower bounds
self.VarMax = np.array([60, 60, 60, 20, 60])    # upper bounds
```

`MaxIt`, `nPop`, and `Run_Time` control the optimizer:

```python
self.MaxIt    = 200  # maximum number of iterations
self.nPop     = 50   # population size (number of candidate solutions)
self.Run_Time = 1    # number of independent optimization runs
```

#### 3.3.3 Load Input

Set `load_type` to choose how the electrical load is provided:

| `load_type` value | Description | Required variable |
|:-----------------:|-------------|-------------------|
| `1` | Hourly load from a CSV file (8,760 rows) | `path_Eload`: path to CSV |
| `2` | Monthly hourly average load | `Monthly_haverage_load`: array of 12 values (kW) |
| `3` | Monthly daily average load | `Monthly_daverage_load`: array of 12 values (kWh/day) |
| `4` | Monthly total load | `Monthly_total_load`: array of 12 values (kWh/month) |
| `5` | Generic profile scaled to monthly totals | `user_defined_load`: array of 12 monthly totals |
| `6` | Annual hourly average | `Annual_haverage_load`: single value (kW) |
| `7` | Annual daily average | `Annual_daverage_load`: single value (kWh/day) |
| `8` | Generic profile scaled to annual total | `Annual_total_load`: single value (kWh/year) |
| `9` | Exact generic load profile (unscaled) | `peak_month`: `'July'` or `'January'` |
| `10` | Generic profile scaled to user daily profile from CSV | `path_Eload_daily`: path to 24-row CSV |

> **Note:** The `content/` folder contains a sample `Eload.csv` for Boston, MA. To use your own load, replace this file or set `path_Eload` to your file path.

#### 3.3.4 Weather Input

`G_type` selects irradiance input, `T_type` selects temperature, and `WS_type` selects wind speed:

| Type flag | Value 1 | Value 2 | Value 3 | Value 4 |
|-----------|---------|---------|---------|---------|
| `G_type` (irradiance) | From NSRDB via SAM API (uses METEO.csv) | User CSV file (plane-of-array) | N/A | N/A |
| `T_type` (temperature) | From NSRDB via SAM API (uses METEO.csv) | User CSV file | Monthly averages (12-value array) | Annual average (single value) |
| `WS_type` (wind speed) | From NSRDB via SAM API (uses METEO.csv) | User CSV file | Monthly averages (12-value array) | Annual average (single value) |

The default `weather_url` points to the bundled `METEO.csv`, which contains sample data for London, Ontario from the National Solar Radiation Database. Key PV geometry parameters:

```python
azimuth = 180   # degrees (180 = south-facing)
tilt    = 33    # degrees (tilt angle of PV modules)
soiling = 5     # percent soiling losses
```

#### 3.3.5 Economic Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `n` | 25 | Project lifetime in years |
| `n_ir_rate` | 2.75 | Nominal discount rate (%) |
| `e_ir_rate` | 2.0 | Expected inflation rate (%) |
| `Budget` | 2,000,000 | Maximum allowable capital cost ($) |
| `Tax_rate` | 0 | Equipment sales tax (%) |
| `RE_incentives_rate` | 30 | Renewable energy incentives / federal tax credit (%) |
| `Pricing_method` | 2 | 1 = top-down (total system $/kW), 2 = bottom-up (itemized costs) |

#### 3.3.6 Component Cost Parameters (Bottom-Up, `Pricing_method = 2`)

| Component | Capital cost param | Replacement param | O&M param | Units |
|-----------|--------------------|-------------------|-----------|-------|
| PV modules | `C_PV = 338` | `R_PV = 338` | `MO_PV = 30.36` | $/kW, $/kW, $/kW/year |
| Inverter | `C_I = 314` | `R_I = 314` | `MO_I = 0` | $/kW, $/kW, $/kW/year |
| Wind turbine | `C_WT = 1200` | `R_WT = 1200` | `MO_WT = 40` | $/kW, $/kW, $/kW/year |
| Diesel generator | `C_DG = 818` | `R_DG = 818` | `MO_DG = 0.016` | $/kW, $/kW, $/op.h |
| Battery | `C_B = 1450` | `R_B = 1450` | `MO_B = 10` | $/kWh, $/kWh, $/kWh/year |
| Heat pump | `C_HP = 109.5` | `R_HP = 109.5` | `MO_HP = 20` | $/1000 BTU/hr, $/1000 BTU/hr, $/year |
| EV battery replacement | N/A | `R_EVB = 27000` | `MO_EV = 0` | $, $/year |

Additional PV engineering costs (bottom-up) are summed into `Engineering_Costs` ($/kW):

```python
self.Fieldwork       = 178   # $/kW
self.Officework      = 696   # $/kW
self.Electrical_BoS  = 333   # $/kW
self.Structrual_BoS  = 237   # $/kW
```

#### 3.3.7 Electricity Tariff Structure

Set `rateStructure` to choose the utility tariff model for grid electricity purchases:

| `rateStructure` | Type | Required parameters |
|:---------------:|------|---------------------|
| `1` | Flat rate | `flatPrice`: single $/kWh value |
| `2` | Seasonal rate | `seasonalPrices [summer, winter]`, `season` [12-value month array] |
| `3` | Monthly rate | `monthlyPrices`: 12-value array ($/kWh per month) |
| `4` | Tiered rate | `tieredPrices`, `tierMax`: tier prices and monthly kWh limits |
| `5` | Seasonal tiered rate | `seasonalTieredPrices [2 seasons x 3 tiers]`, `seasonalTierMax`, `season` |
| `6` | Monthly tiered rate | `monthlyTieredPrices [12 x 3]`, `monthlyTierLimits [12 x 3]` |
| `7` | Time-of-Use (TOU) | `onPrice`, `midPrice`, `offPrice [summer, winter]`, `onHours`, `midHours`, `season`, `treat_special_days_as_offpeak` |
| `8` | Ultra-Low TOU (ULO) | `onPrice`, `midPrice`, `offPrice`, `ultraLowPrice [summer, winter]`, `onHours`, `midHours`, `ultraLowHours`, `season`, `treat_special_days_as_offpeak` |

Grid sell price structure is set separately with `sellStructure`:

| `sellStructure` | Description | Parameter |
|:---------------:|-------------|-----------|
| `1` | Flat sell price | `Csell = np.full(8760, price)`: one price all year |
| `2` | Monthly sell price | `monthlysellprices`: 12-value array ($/kWh per month) |
| `3` | Same as buy price | `Csell = Cbuy`: NEM at retail rate |

Net metering is controlled by `NEM` (`1` = enabled) and `NEM_fee` (one-time connection fee in $). When `NEM = 1`, surplus generation credits are reconciled annually.

#### 3.3.8 Natural Gas Tariff (for Heat Pump)

When `HP = 1` and a natural gas furnace baseline is being compared, set `rateStructure_NG` to one of 8 structures analogous to the electricity rate structures. The NG price is entered in the user's native unit ($/m³, $/therm, or $/kWh) and SAMA converts internally using `NG_energycontent` (10.81 kWh/m³) and `Furnace_eff` (0.94).

#### 3.3.9 System Constraints

| Parameter | Default | Description |
|-----------|---------|-------------|
| `LPSP_max_rate` | 0.000999 | Maximum Loss of Power Supply Probability (LPSP) as a percentage. The system must meet at least (100 minus `LPSP_max_rate`)% of demand. |
| `RE_min_rate` | 50 | Minimum renewable energy fraction required (%). System must have at least this fraction from PV and WT. |
| `EM` | 0 | Objective mode. 0 = minimize NPC only. 1 = minimize NPC + Levelized Emissions (LEM). |
| `Budget` | 2,000,000 | Maximum total capital cost ($). Infeasible solutions above this are penalized. |
| `cap_option` | 2 | Capacity sizing constraint. 1 = explicit cap in kW (`cap_size`). 2 = size to annual load. 3 = rooftop area limit. 4 = no limit. |

### 3.4 Running the Optimization

After configuring `Input_Data.py`, run one of the optimizer scripts directly. In PyCharm, right-click the script and choose Run, or use the terminal.

#### 3.4.1 Using `run_sama_optimized.py` (recommended for raw-code users)

```bash
python run_sama_optimized.py --algorithm pso
python run_sama_optimized.py --algorithm ade
python run_sama_optimized.py --algorithm abc
python run_sama_optimized.py --algorithm gwo
```

This script loads `sama_config_COMPLETE_HYBRID.yaml` (if present) and applies it to `InData` at runtime, then runs the chosen algorithm.

#### 3.4.2 Running optimizer modules directly

```bash
python sama/optimizers/pso.py        # Particle Swarm Optimization
python sama/optimizers/ade.py        # Advanced Differential Evolution
python sama/optimizers/run_abc.py    # Artificial Bee Colony
python sama/optimizers/gwo.py        # Grey Wolf Optimizer
```

#### 3.4.3 Using the `examples/` folder

The `examples/optimization_case.py` file contains a complete working example with all 8 electricity rate structures and all 6 natural gas rate structures fully commented. It defines a `Config` class with all parameters and runs the full optimization workflow.

```bash
python examples/optimization_case.py
```

---

## 4. Method 2: Using the Python Package (pip install samapy)

The SAMA pip package (SAMA Py Package) is available through the Python Package Index (PyPI). It provides two console commands, **`samapy-config`** and **`samapy-run`**, that guide you through configuration and run the optimization without editing any source files. This is the recommended approach for most users.

### 4.1 Installation

Install SAMA from PyPI (the SAMA Py Package):

```bash
pip install samapy
```

Verify the installation:

```bash
samapy-config --help
samapy-run --help
```

> **Note:** Python 3.9 or higher is required. On Windows, ensure Python is in your PATH. All dependencies (`numpy`, `pandas`, `scipy`, `numba`, `matplotlib`, `openpyxl`, `seaborn`, `numpy-financial`, `questionary`, `PyYAML`) are installed automatically.

### 4.2 Step 1: Run the Configuration Wizard

In your working directory (the folder where you want results to be saved), run:

```bash
samapy-config
```

The wizard launches an interactive terminal interface with **18 sections** that guide you through every parameter category. You answer questions using your keyboard; arrow keys to select and Enter to confirm. At the end of each section the wizard saves progress and moves to the next.

| Wizard Section | Parameters Covered |
|----------------|-------------------|
| **Section 1:** System Components | Which components to include: PV, WT, DG, Battery (lead-acid or Li-ion), Grid, Heat Pump (Bosch or Goodman), EV |
| **Section 2:** Optimization Settings | Algorithm (PSO/ADE/ABC/GWO), population size, iterations, run count, design variable bounds (VarMin, VarMax) |
| **Section 3:** Project and Calendar | Project lifetime (n), simulation year, public holiday list |
| **Section 4:** Electrical Load | Load type selection (1–10), CSV file paths or manual values, previous-year load for NEM calculations |
| **Section 5:** Thermal Load | Heat pump thermal load source (house_load.xlsx or manual) |
| **Section 6:** Weather and Irradiance | METEO.csv path or manual weather input, PV tilt/azimuth/soiling |
| **Section 7:** Constraints | LPSP maximum, renewable energy minimum fraction, budget limit, capacity constraint type |
| **Section 8:** Financial | Discount rates, inflation rate, tax rate, RE incentives/federal tax credit |
| **Section 9:** PV and Inverter | Pricing method, module efficiency, NOCT parameters, lifetimes, capital/replacement/O&M costs |
| **Section 10:** Wind Turbine | Hub height, cut-in/rated/cut-out speeds, power law coefficient, costs |
| **Section 11:** Diesel Generator | Rated capacity, fuel curve (a, b), minimum load ratio, emissions, fuel cost, lifetime |
| **Section 12:** Battery Storage | Li-ion or lead-acid selection, SOC limits, round-trip efficiency, capacity, lifetime throughput, costs |
| **Section 13:** Heat Pump | HP brand (Bosch or Goodman), rated size, lifetime, costs |
| **Section 14:** Electric Vehicle | Battery capacity, SOC limits, charging power, range, daily trip, departure/arrival times, degradation |
| **Section 15:** Grid and NEM | Net metering flag, capacity option (cap_size), grid tax, annual service charge, sell structure |
| **Section 16:** Electricity Rate Structure | Rate type (1–8), prices, TOU hours, seasons, holidays treatment |
| **Section 17:** Natural Gas Rate Structure | NG unit (m³/therms/kWh), rate type (1–8), NG prices, escalation |
| **Section 18:** Output Settings | Output directory, compare-with-grid scenario flag |

When the wizard completes, it saves a file called `sama_config_COMPLETE.yaml` in your working directory. This YAML file contains all your parameters and can be edited manually or re-run through the wizard at any time.

### 4.3 Step 2: Run the Optimization

With `sama_config_COMPLETE.yaml` in your working directory, run:

```bash
samapy-run
```

SAMA will automatically find the config file, load all parameters, and run the optimization algorithm specified in the config (`optimization_algorithm` key). Additional options are available:

#### 4.3.1 `samapy-run` command options

| Option | Description | Example |
|--------|-------------|---------|
| `-c` / `--config` | Path to config YAML (default: `sama_config_COMPLETE.yaml` in cwd) | `samapy-run -c my_project.yaml` |
| `-a` / `--algorithm` | Override the algorithm from the config file | `samapy-run -a pso` |
| `--output` | Override the output directory from the config | `samapy-run --output results/run1` |
| `--dry-run` | Validate the config file without running optimization | `samapy-run --dry-run --verbose` |
| `--no-gui` | Disable matplotlib GUI (use Agg backend for servers) | `samapy-run --no-gui` |
| `--verbose` | Show full parameter log and config summary | `samapy-run --verbose` |

The `optimization_algorithm` key in the YAML accepts: `pso`, `ade`, `abc`, or `gwo`.

### 4.4 Step 3: View Results

Results are saved to the output directory specified in the config (default: `sama_outputs/` in your working directory). The folder is organized as:

```
sama_outputs/
    Optimization.png          convergence curve (best cost vs. iteration)
    figs/                     all chart PNG/SVG files
    data/                     CSV data files
```

#### 4.4.1 Chart files generated in `sama_outputs/figs/`

| File | Description |
|------|-------------|
| `Cash_Flow.svg` | 25-year cash flow breakdown (capital, replacement, O&M, fuel, grid) |
| `Cash Flow_ADV.png` | Advanced cash flow with grid comparison (when `compare_with_grid = 1`) |
| `Multiple_Cash_Flow_ADV.png` | Multi-run cash flow comparison (when `Run_Time > 1`) |
| `Energy Distribution.png` | Annual energy supply breakdown by source (PV, WT, DG, battery, grid) |
| `Battery State of Charge.png` | Hourly battery SOC profile over the year |
| `Grid Interconnection.png` | Hourly grid import and export over the year |
| `Specific day results.png` | Detailed 24-hour power flow for a representative day |
| `Daily-Monthly-Yearly average cost of energy system.png` | Cost comparison across daily, monthly, and annual timescales |
| `Daily-Monthly-Yearly average hourly cost of connecting to the grid.png` | Hourly grid cost patterns by season |
| `Daily-Monthly-Yearly average cost of only grid-connected system.png` | Baseline grid-only cost for comparison |
| `Grid Hourly Cost.png` | Hourly grid electricity cost profile |
| `Daily-Monthly-Yearly average earning Sell to the Grid.png` | Revenue from grid exports (if NEM enabled) |
| `Energy Balance.png` | Hourly energy balance (supply vs. demand) |
| `electricity_comparison.png` | SAMA system vs. grid-only electricity cost comparison |
| `EV Energy.png` | EV charging and discharging energy over the year |
| `EV Sp Results 1.png` | EV-specific results chart 1 (when `EV = 1`) |
| `EV Sp Results 2.png` | EV-specific results chart 2 (when `EV = 1`) |
| `hp_ambient_conditions.png` | Ambient temperature and pressure conditions for heat pump (when `HP = 1`) |
| `hp_heating_performance.png` | Heat pump heating power and COP (when `HP = 1`) |
| `hp_cooling_performance.png` | Heat pump cooling power and COP (when `HP = 1`) |
| `hp_cop_vs_temp.png` | COP as a function of ambient temperature (when `HP = 1`) |
| `hp_monthly_summary.png` | Monthly heat pump energy and COP summary (when `HP = 1`) |
| `Optimization.png` | Optimizer convergence curve (saved in `sama_outputs/` root) |

#### 4.4.2 Data files generated in `sama_outputs/data/`

| File | Description |
|------|-------------|
| `Outputforplotting.csv` | Complete hourly time-series: PV output, WT output, battery SOC, DG power, grid import/export, load, EV charge/discharge, and all cost variables for all 8,760 hours |
| `cash_flow_data.csv` | Annual cash flow breakdown over the 25-year project lifetime |

### 4.5 Using the Python API

After `pip install samapy`, SAMA can also be used directly in Python scripts without the command-line interface:

#### 4.5.1 Basic API workflow

```python
from samapy.core.Input_Data import InData
from samapy.cli.config_loader import load_config, apply_config

# Load a YAML config produced by samapy-config
config = load_config('sama_config_COMPLETE.yaml')

# Apply all parameters to the InData singleton
apply_config(config)

# Run the optimizer
from samapy.optimizers.swarm import Swarm
swarm = Swarm()
swarm.optimize()
```

#### 4.5.2 Available optimizer classes

| Algorithm | Import | Class |
|-----------|--------|-------|
| Particle Swarm Optimization | `from samapy.optimizers.swarm import Swarm` | `Swarm()` |
| Advanced Differential Evolution | `from samapy.optimizers.AdvancedDifferentialEvolution import AdvancedDifferentialEvolution` | `AdvancedDifferentialEvolution()` |
| Artificial Bee Colony | `from samapy.optimizers.ArtificialBeeColony import ImprovedArtificialBeeColony` | `ImprovedArtificialBeeColony()` |
| Grey Wolf Optimizer | `from samapy.optimizers.GreyWolfOptimizer import GreyWolfOptimizer` | `GreyWolfOptimizer()` |

#### 4.5.3 Config loader helper functions

| Function | Description |
|----------|-------------|
| `load_config(path)` | Load a YAML config file and return it as a Python dictionary |
| `apply_config(config)` | Apply a config dictionary to the InData singleton and return the fully reconfigured InData instance |
| `save_config(config, path)` | Save a config dictionary to a YAML file |
| `merge_configs(*paths)` | Merge multiple YAML config files into one dictionary (later files override earlier ones) |

#### 4.5.4 Important: Package Name vs Import Name

The SAMA Python package is published on PyPI under the distribution name **`samapy`**. This is the name you use with pip to install the package. However, once installed, the package is imported in Python code using the name **`sama`** (all lowercase). These are two different names for the same package:

**Install with pip** (use the PyPI distribution name):
```bash
pip install samapy
```

**Import in Python scripts** (use the internal package name):
```python
import samapy
```

**Command line tools** installed automatically:
```bash
samapy-config
samapy-run
```

When searching for the package on the PyPI website, look for `samapy`. When writing Python scripts that use SAMA, always write `import samapy`.

### 4.6 Using Your Own Custom Data Files

SAMA ships with bundled default data files for London, Ontario (weather, electrical load, and heat pump data). When you run `samapy-config` and provide your own file paths, the wizard automatically copies those files into the local `samapy/content/` folder in your working directory. SAMA reads from this local folder instead of the bundled defaults.

You can also copy your custom files directly into the `samapy/content/` folder yourself, and SAMA will use them automatically when you run `samapy-run`. The `samapy/content/` folder is created in your working directory the first time you run `samapy-config` or `samapy-run`.

The expected file names and formats are:

- **`Eload.csv`**, Hourly electrical load profile. One column, 8,760 rows, no header. Values in kW.
- **`METEO.csv`**, SAM NSRDB weather file for your location. Two header rows followed by 8,760 hourly data rows. Must include columns: GHI, DNI, DHI, Temperature, Wind Speed, Pressure. Download from https://nsrdb.nrel.gov/
- **`house_load.xlsx`**, Building thermal load. Excel file with heating load (Hload) in column 2 and cooling load (Cload) in column 3, 8,760 rows each, in kW.
- **`Irradiance.csv`**, Only needed when `G_type = 2` (manual plane-of-array irradiance). One column, 8,760 rows, no header. Values in W/m².

If a file is not present in the local `samapy/content/` folder, SAMA automatically falls back to the bundled default. You only need to provide files you want to customize.

### 4.7 Typical Workflow for a New Project Location

1. Create an empty project folder, for example `C:\Projects\MyCity_SAMA`
2. Download your `METEO.csv` weather file from https://nsrdb.nrel.gov/ for your project location.
3. Prepare your `Eload.csv` file with 8,760 rows of hourly electrical load data in kW.
4. If modeling a heat pump, prepare `house_load.xlsx` with Hload and Cload columns.
5. Open a terminal in your project folder and run `samapy-config`. The wizard will ask for paths to your data files and copy them into `samapy/content/` automatically.
6. When the wizard finishes, `sama_config_COMPLETE.yaml` is saved in your project folder. You can open it in any text editor to review or manually adjust settings.
7. Run `samapy-run` to start the optimization. Results are saved in `sama_outputs/` inside your project folder.

### 4.8 Understanding the Working Directory

SAMA creates all its folders (`samapy/content/`, `sama_inputs/`, `sama_outputs/`) relative to the folder where you run the `samapy-config` and `samapy-run` commands. This folder is called the **working directory**. Always run both commands from the same project folder so that the configuration file, custom data files, and results all stay together in one place.

> **On Windows:** navigate to your project folder in File Explorer, click the address bar, type `cmd`, and press Enter. This opens a Command Prompt already pointed at that folder.

### 4.9 What to Expect on the First Run

The very first time you run **`samapy-run`** after installation, SAMA uses Numba to compile the energy management system (EMS) functions. This compilation is a one-time process and can take several minutes. You will see messages in the terminal while this happens. Subsequent runs will be significantly faster because the compiled code is cached on your computer.

For a typical run with `MaxIt=200` and `nPop=50`, expect the optimization itself to take between **5 and 30 minutes** depending on your hardware and which components (EV, heat pump) are enabled.

---

## 5. Method 3: Windows .exe Application (Alpha)

A Windows executable (`.exe`) version of SAMA is currently in the alpha development phase and is available on request from the developers. This version provides a graphical user interface (GUI) that does not require Python to be installed on the user's computer. It is intended for non-technical users who prefer a point-and-click experience.

Because this version is in the alpha phase, some features may be incomplete or subject to change. Users are encouraged to report any issues to the development team.

To request the Windows `.exe` version, contact the FAST research group at Western University. Contact information is available at the GitHub repository: https://github.com/Sas1997/SAMA

> **Windows .exe Application (Alpha Development Phase)**
> - The `.exe` version is **not** distributed via PyPI.
> - It must be obtained directly from the developers.
> - No Python installation is required to run the `.exe`.
> - It uses the same algorithms and produces the same outputs as the Python package.
> - This version is currently in the alpha development phase. Features may be incomplete.

---

## 6. The Configuration File (`sama_config_COMPLETE.yaml`)

Whether you use the wizard (`samapy-config`) or edit it manually, the configuration file is a standard YAML text file. You can open it in any text editor. Below is a complete reference of all keys.

### 6.1 File Structure Overview

```yaml
input_directory: C:\path\to\project   # working directory (used by raw code)
output_directory: C:\path\to\sama_outputs  # results output folder
optimization_algorithm: abc            # pso | ade | abc | gwo

# Optimization settings
MaxIt: 200
nPop: 50
Run_Time: 1
VarMin: [0.0, 0.0, 0.0, 0.0, 0.0]
VarMax: [60.0, 60.0, 60.0, 20.0, 60.0]

# System components (1=included, 0=excluded)
PV: 1   WT: 1   DG: 1   Bat: 1   Grid: 1   HP: 1   EV: 1
Li_ion: 1   Lead_acid: 0
NEM: 1

# Project calendar
n: 25        # project lifetime (years)
year: 2023   # simulation year
holidays: [1, 51, 97, ...]  # day-of-year numbers for public holidays
```

### 6.2 Key Parameter Reference

#### Optimization

| YAML key | Type | Default | Description |
|----------|------|---------|-------------|
| `MaxIt` | int | 200 | Maximum optimizer iterations |
| `nPop` | int | 50 | Population size (number of candidate solutions) |
| `Run_Time` | int | 1 | Independent optimization runs (best result kept) |
| `VarMin` / `VarMax` | list[5] | `[0,0,0,0,0]` / `[60,60,60,20,60]` | Lower/upper bounds for [Npv, Nwt, Nbat, N_DG, Cn_I] |
| `Cash_Flow_adv` | int | 0 | 0 = standard cash flow; 1 = advanced multi-run cash flow chart |

#### Calendar

| YAML key | Type | Default | Description |
|----------|------|---------|-------------|
| `n` | int | 25 | Project lifetime in years |
| `year` | int | 2023 | Simulation year (affects day-of-week and leap year) |
| `holidays` | list[int] | `[1,51,97,...]` | Day-of-year numbers for public holidays (affects TOU pricing) |

#### Load

| YAML key | Type | Default | Description |
|----------|------|---------|-------------|
| `load_type` | int | 1 | Electrical load input mode (1–10, see Section 3.3.3) |
| `load_previous_year_type` | int | 1 | Previous year load mode for NEM credit calculation (1–11) |
| `Tload_type` | int | 1 | Thermal load input type (1 = from house_load.xlsx) |

#### Weather

| YAML key | Type | Default | Description |
|----------|------|---------|-------------|
| `G_type` | int | 1 | Irradiance input mode (1=NSRDB/SAM, 2=CSV) |
| `T_type` | int | 1 | Temperature input mode (1=NSRDB/SAM, 2=CSV, 3=monthly avg, 4=annual avg) |
| `WS_type` | int | 1 | Wind speed input mode (1=NSRDB/SAM, 2=CSV, 3=monthly avg, 4=annual avg) |
| `azimuth` | float | 180.0 | PV panel azimuth angle (degrees, 180 = south-facing) |
| `tilt` | float | 33.0 | PV panel tilt angle (degrees) |
| `soiling` | float | 5.0 | PV soiling losses (%) |
| `weather_url` | str | METEO.csv | Path to NSRDB/SAM meteorological CSV file |

#### Constraints

| YAML key | Type | Default | Description |
|----------|------|---------|-------------|
| `LPSP_max_rate` | float | 0.000999 | Maximum LPSP (% of annual demand that can be unmet) |
| `RE_min_rate` | float | 50.0 | Minimum renewable energy fraction required (%) |
| `EM` | int | 0 | 0 = minimize NPC; 1 = minimize NPC + LEM (emissions) |
| `Budget` | float | 2,000,000 | Maximum allowable capital cost ($) |
| `cap_option` | int | 1 | 1=size cap (kW), 2=annual load, 3=rooftop area, 4=no limit |
| `cap_size` | float | 15.0 | PV system size cap in kW (used when `cap_option=1`) |

#### Financial

| YAML key | Type | Default | Description |
|----------|------|---------|-------------|
| `n_ir_rate` | float | 2.75 | Nominal discount rate (%) |
| `e_ir_rate` | float | 2.0 | Expected inflation rate (%) |
| `Tax_rate` | float | 0.0 | Equipment sales tax (%) |
| `RE_incentives_rate` | float | 30.0 | Federal/provincial RE incentives / tax credit (%) |
| `Pricing_method` | int | 2 | 1=top-down (`Total_PV_price` $/kW), 2=bottom-up (itemized) |

#### Grid

| YAML key | Type | Default | Description |
|----------|------|---------|-------------|
| `Grid_sale_tax_rate` | float | 13.0 | Electricity sales tax (%) |
| `Grid_Tax_amount` | float | 0.0353 | Fixed grid tax in $/kWh (e.g. delivery charge component) |
| `Grid_credit` | float | 0.0 | Annual grid credit to customers in $ (e.g. rebate) |
| `Grid_escalation_rate` | list[12] | 2.0 per year | Annual electricity price escalation rate per year (%) |
| `NEM` | int | 1 | 1 = net metering enabled (annual credit reconciliation) |
| `NEM_fee` | float | 0.0 | One-time NEM interconnection setup fee ($) |
| `Monthly_fixed_charge_system` | int | 1 | 1=flat service charge, 2=tiered service charge |
| `SC_flat` | float | 27.16 | Flat monthly service charge ($) |
| `rateStructure` | int | 7 | Electricity rate type (1–8, see Section 3.3.7) |
| `sellStructure` | int | 1 | Grid sell price structure (1=flat, 2=monthly, 3=same as buy) |
| `Pbuy_max` | float | 50.0 | Maximum grid import power (kW) |
| `Psell_max` | float | 50.0 | Maximum grid export power (kW) |

#### Natural Gas

| YAML key | Type | Default | Description |
|----------|------|---------|-------------|
| `NG_energycontent` | float | 10.81 | Energy content of 1 m³ of natural gas (kWh/m³) |
| `Furnace_eff` | float | 0.94 | Efficiency of natural gas furnace (fraction) |
| `NG_unit` | str | therms | Unit of NG prices entered: `m3`, `therms`, or `kwh` |
| `rateStructure_NG` | int | 1 | NG rate type (1–8) |
| `SC_flat_NG` | float | 18.59 | Flat monthly NG service charge ($) |

---

## 7. Optimization Algorithms

SAMA includes four metaheuristic optimization algorithms. All four optimize the same objective (NPC, or NPC+LEM when `EM=1`) over the same 5-variable design space. The choice of algorithm does not affect *what* is being optimized, only *how* the search space is explored.

### 7.1 Particle Swarm Optimization (PSO)

PSO simulates a swarm of candidate solutions (particles) that move through the search space. Each particle adjusts its velocity based on its own best-known position and the swarm's global best position. Key parameters:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `w` | 1.0 | Inertia weight: controls how much a particle keeps its current velocity |
| `wdamp` | 0.99 | Inertia weight damping ratio: `w` is multiplied by `wdamp` each iteration |
| `c1` | 2.0 | Personal learning coefficient: attraction toward particle's own best position |
| `c2` | 2.0 | Global learning coefficient: attraction toward swarm's global best position |

### 7.2 Advanced Differential Evolution (ADE)

ADE is a population-based algorithm that creates new candidate solutions by combining existing ones using differential mutation and crossover. SAMA's ADE implementation uses multiple DE strategies with adaptive parameters:

- Five mutation strategies: `DE/rand/1`, `DE/best/1`, `DE/current-to-best/1`, `DE/rand/2`, `DE/best/2`
- Adaptive scaling factor F and crossover probability CR, tuned by success rate and iteration progress
- Heterogeneous initialization (random, center-biased, boundary-biased, Sobol-like) for diverse coverage

| Parameter | Default | Description |
|-----------|---------|-------------|
| `F_min` / `F_max` | 0.1 / 0.9 | Minimum/maximum scaling factor for mutation |
| `CR_min` / `CR_max` | 0.1 / 0.9 | Minimum/maximum crossover probability |

### 7.3 Improved Artificial Bee Colony (ABC)

ABC mimics the foraging behavior of honey bees. It has three phases: employed bees, onlooker bees, and scout bees, each exploring and exploiting promising regions of the search space. SAMA's ABC uses multi-dimensional perturbations and guided reinitialization of stagnant solutions.

| Parameter | Default | Description |
|-----------|---------|-------------|
| `maxTrials` | 15 | Maximum number of trials before a food source is abandoned and re-scouted |
| `modification_rate` | 0.8 | Probability of modifying multiple dimensions simultaneously |
| `initial_search_radius` | 0.5 | Search radius at the start of optimization (fraction of variable range) |
| `final_search_radius` | 0.1 | Search radius at the end of optimization |

### 7.4 Grey Wolf Optimizer (GWO)

GWO mimics the social hierarchy and hunting behavior of grey wolves. Solutions are classified as alpha (best), beta, delta, and omega wolves. The pack converges on prey (optimal solution) guided by alpha, beta, and delta wolves. The parallel variant (`ParallelGreyWolfOptimizer`) uses Python multiprocessing for faster evaluation when `Run_Time > 1`.

### 7.5 Choosing an Algorithm

All four algorithms are suitable for this problem. In practice, for the default 5-variable search space with `MaxIt=200` and `nPop=50`, any algorithm will find good solutions. ADE and ABC tend to explore more thoroughly; PSO and GWO converge faster. For a rigorous comparison, use `Run_Time > 1` to average across multiple independent runs.

---

## 8. Energy Management System (EMS)

For each candidate system design, SAMA simulates 8,760 hours of operation using an hourly dispatch model. The EMS decides, for every hour, how much power each component supplies or consumes.

### 8.1 Base Dispatch (`EMS.py`)

The base EMS uses Numba JIT compilation for fast execution. In each hour it performs marginal-cost-based dispatch:

1. Calculate available PV and wind power from irradiance and wind speed.
2. Calculate net demand (load minus PV minus wind output).
3. If net demand is positive (deficit): draw from battery, then diesel generator, then grid, based on relative hourly cost.
4. If net demand is negative (surplus): charge battery, then export to grid if NEM is enabled.
5. Track battery SOC, respect `SOC_min`, `SOC_max`, and charge/discharge current limits.
6. Track any unmet energy (`Ens`) toward LPSP calculation.

### 8.2 EV Smart Charging and V2X (`EMS_EV.py`)

When `EV = 1`, the `EMS_EV` module adds vehicle-to-home (V2H) and vehicle-to-grid (V2G) capabilities. Key behaviors:

- EV is only available to charge or discharge when it is at home (`EV_p = 1`, determined by departure time `Tout`, arrival time `Tin`, and holiday schedule).
- **Departure planning:** the EV must reach `SOC_dep` before `Tout` each day.
- **Lookahead arbitrage:** SAMA looks ahead at future grid prices to decide whether to discharge the EV battery now (selling to grid or powering the home) or wait. The decision accounts for battery wear cost and round-trip efficiency losses.
- **Safety buffer:** a reserve SOC is maintained based on historical departure patterns and load variability.
- **V2H discharge** is used when it is economically advantageous (current buy price greater than battery wear cost plus discharge losses).

Key EV parameters set in `Input_Data.py` or the wizard:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `C_ev` | 82 kWh | EV battery total capacity |
| `SOCe_min` / `SOCe_max` | 0.03 / 0.97 | Minimum and maximum usable SOC fraction |
| `Pev_max` | 11 kW | Maximum charge and discharge power |
| `Range_EV` | 468 km | EV full-charge driving range |
| `Daily_trip` | 68 km | Average daily distance driven |
| `SOC_dep` | 0.85 | Required SOC at departure time |
| `Tin` / `Tout` | 17 / 8 | Arrival hour (17 = 5 PM) and departure hour (8 = 8 AM) |
| `n_e` | 0.9 | EV battery round-trip charge efficiency |
| `L_EV_dis` | 400,000 km | EV battery lifetime in total km driven |
| `treat_special_days_as_home` | false | If true, holidays are treated as home days for EV presence |

### 8.3 Heat Pump Energy Attribution (`EMS_HP.py`)

When `HP = 1`, `EMS_HP` post-processes the dispatch results to attribute the electricity consumed by the heat pump to heating versus cooling loads. Two methods are used: proportional allocation (based on hourly heating and cooling fractions) and marginal-cost-based assignment. This enables detailed multi-fuel cost accounting that separates electricity costs for space heating from space cooling.

---

## 9. Component Models

### 9.1 Solar PV

Hourly PV power output uses a temperature-dependent efficiency model with NOCT-based cell temperature correction:

- Cell temperature is calculated from ambient temperature, NOCT, and irradiance.
- Module efficiency is adjusted for cell temperature using `Tcof` (temperature coefficient, default −0.3 %/°C).
- Output is multiplied by the derating factor `fpv` (default 0.9) for real-world losses.
- The default module rated power is `Ppv_r = 1 kW`; the optimizer returns the number of modules `Npv`.

### 9.2 Wind Turbine

- Hub-height wind speed is calculated from anemometer height `h0` to hub height `h_hub` using a power-law profile with friction coefficient `alfa_wind_turbine` (default 0.14).
- Power output uses a piecewise model: zero below cut-in speed (`v_cut_in = 2.5 m/s`), linear ramp to rated power between cut-in and rated speed (`v_rated = 9.5 m/s`), rated power between rated and cut-out (`v_cut_out = 25 m/s`), and zero above cut-out.
- The default rated power is `Pwt_r = 1 kW`; the optimizer returns the number of turbines `Nwt`.

### 9.3 Battery Storage

SAMA supports two battery chemistries:

- **Li-ion:** idealized model with fixed round-trip efficiency `ef_bat_Li` (default 0.90), maximum charge current `Ich_max_Li_ion`, maximum discharge current `Idch_max_Li_ion`, lifetime throughput `Q_lifetime_Li` (default 3,000 kWh per unit), and nominal capacity `Cnom_Li`.
- **Lead-acid:** KiBaM (Kinetic Battery Model) with capacity ratio `c` and storage rate constant `k_lead_acid`, maximum charge current `Ich_max_leadacid`, and lifetime throughput `Q_lifetime_leadacid` (default 8,000 kWh per unit).

Both types use `SOC_min` (default 0.1) and `SOC_max` (default 1.0) to bound the state of charge. The battery rated capacity `Cbt_r` is derived from the cell voltage and nominal capacity.

### 9.4 Diesel Generator

Fuel consumption is modeled as an affine function of output power: `F = a * Prated + b * P_output` (L/hr). Parameters `a = 0.4388` and `b = 0.1097` are default values. The minimum loading ratio `LR_DG` (default 0.25) prevents the generator from running at very low fractions of rated capacity. Emissions per litre of fuel are set by `CO2`, `CO`, `NOx`, and `SO2`.

### 9.5 Heat Pump Models

SAMA includes black-box models for two heat pump brands:

- **Goodman** (`BB_HP_Goodman.py`): Uses manufacturer data to compute hourly electric consumption for space heating and cooling as a function of outdoor ambient temperature and building thermal load (`Hload`, `Cload`). Returns COP curves for heating and cooling modes.
- **Bosch** (`BB_HP_Bosch.py`): Same structure as the Goodman model but using Bosch manufacturer data. Select using `HP_brand = 'Bosch'` or `HP_brand = 'Goodman'`.

Both models read lookup data from the `content/` folder (`HP_Bosch/` and `HP_Goodman/` subdirectories containing Excel data files).

---

## 10. Financial Model

### 10.1 Net Present Cost (NPC)

NPC is the total 25-year lifecycle cost of the hybrid energy system in today's dollars. It includes:

- **Initial capital cost** (`I_Cost`): PV, WT, DG, battery, inverter, charger, engineering costs, heat pump, EV battery, NEM connection fee, reduced by RE incentives and multiplied by `(1 + System_Tax)`.
- **Replacement costs** (`R_Cost`): scheduled component replacements over the project lifetime, discounted to present value.
- **O&M costs** (`MO_Cost`): annual operations and maintenance costs for all components, escalated and discounted.
- **Fuel cost** (`C_Fu`): diesel generator fuel cost, escalated by `C_fuel_adj_rate` and discounted.
- **Grid electricity cost** (`Grid_Cost_net`): annual grid purchases at tariff rates, escalated by `Grid_escalation_rate` and discounted. NEM credits and grid credits are subtracted.
- **Salvage value**: remaining value of components at end of project life.

```
NPC = (I_Cost + sum(R_Cost) + sum(MO_Cost) + sum(C_Fu) - sum(Salvage)) * (1 + System_Tax)
      + sum(Grid_Cost_net) + sum(gHeating_Cost)
```

### 10.2 Levelized Cost of Energy (LCOE)

`LCOE = CRF * NPC / E_tot`, where CRF is the capital recovery factor derived from the real discount rate `ir` and project lifetime `n`, and `E_tot` is the total discounted energy supplied over the project lifetime (including EV charging and thermal energy from the heat pump).

### 10.3 Levelized Emissions (LEM)

LEM is computed when `EM = 1`. It is the total lifecycle emissions (from diesel generation and grid-sourced electricity) normalized by total supplied energy. Emission factors `CO2`, `CO`, `NOx`, `SO2` (kg/L of diesel) and `E_CO2`, `E_SO2`, `E_NOx` (kg/kWh of grid electricity) are user-specified.

### 10.4 25-Year Cash Flow Projection

The financial model projects cash flows year by year over the `n`-year project lifetime. Grid electricity prices are escalated using `Grid_escalation_rate` (array of 25 annual percentages). Natural gas prices are escalated using `Grid_escalation_rate_NG`. Component replacements are scheduled at multiples of their lifetimes (`L_PV`, `L_WT`, `L_B`, `L_I`, `L_CH`, `L_HP`, `L_EV`). The `Advanced_multi_cashflow` module generates multi-run comparison charts when `Cash_Flow_adv = 1` and `Run_Time > 1`.

---

## 11. Bundled Data Files (`content/` folder)

The following data files are bundled with the SAMA package in the `samapy/content/` directory and can be accessed via `get_content_path()`:

| File | Description |
|------|-------------|
| `Eload.csv` | Sample hourly electrical load profile (8,760 rows, kW) for a representative building in London, Ontario |
| `Eload_daily.csv` | Sample daily electrical load profile (24 rows) for generic load scaling |
| `METEO.csv` | SAM NSRDB meteorological data for London, Ontario. Columns include GHI, DNI, DHI, temperature, wind speed, and pressure |
| `Irradiance.csv` | Sample hourly plane-of-array (POA) irradiance CSV (8,760 rows, W/m²) for manual `G_type=2` input |
| `Temperature.csv` | Sample hourly ambient temperature CSV (8,760 rows, °C) for manual `T_type=2` input |
| `WSPEED.csv` | Sample hourly wind speed CSV (8,760 rows, m/s) for manual `WS_type=2` input |
| `Battery.csv` | Battery lookup data |
| `Data.csv` | General component reference data |
| `Generic_load_JulyP.csv` | Normalized generic hourly load profile with July peak; used when `load_type = 5/8/9/10` |
| `Generic_load_JanuaryP.csv` | Normalized generic hourly load profile with January peak; used when `load_type = 5/8/9/10` |
| `house_load.xlsx` | Sample building heating (Hload) and cooling (Cload) thermal load (8,760 rows, kW each) for London, Ontario |
| `HP_Bosch/` | Bosch heat pump manufacturer performance data files (6 Excel files and weather data) |
| `HP_Goodman/` | Goodman heat pump manufacturer performance data files (8 Excel files and weather data) |

To use your own weather or load data, either replace these files in place or specify alternative file paths in `Input_Data.py` (e.g. `path_Eload`, `path_G`, `path_T`, `path_WS`, `weather_url`) or in the config YAML.

---

## 12. Input Data Formats

### 12.1 Electrical Load CSV (`Eload.csv`)

- One column, 8,760 rows (no header).
- Each row is the hourly average electrical load in kW for that hour of the year.
- Hour 0 = midnight on January 1; hour 8,759 = 11 PM on December 31.

### 12.2 METEO.csv (SAM NSRDB format)

- Two header rows followed by hourly data rows (8,760 rows).
- Must include columns: `GHI`, `DNI`, `DHI`, `Temperature`, `Wind Speed`, `Pressure`.
- Downloaded from NSRDB (https://nsrdb.nrel.gov/) or exported from SAM (System Advisor Model).

### 12.3 Plane-of-Array Irradiance CSV (`Irradiance.csv`)

- One column, 8,760 rows (no header).
- Each row is the hourly POA irradiance in W/m².
- Used when `G_type = 2`.

### 12.4 `house_load.xlsx` (Thermal Load)

- Excel file with at least 3 columns.
- Column index 1 (second column): hourly heating load `Hload` in kW (8,760 rows).
- Column index 2 (third column): hourly cooling load `Cload` in kW (8,760 rows).

---

## 13. Troubleshooting

| Error / Symptom | Cause | Fix |
|-----------------|-------|-----|
| `ModuleNotFoundError: No module named 'sama'` | SAMA not installed in current Python environment | Run `pip install samapy` (or `pip install -e .` from project root) |
| `command not found: samapy-config` | `pip install` did not add scripts to PATH | On Windows: restart terminal. On Linux/Mac: add `~/.local/bin` to PATH |
| `BackendUnavailable: Cannot import setuptools.backends.legacy` | Old version of setuptools | The `pyproject.toml` uses `setuptools.build_meta`. Ensure the correct `pyproject.toml` is present. |
| `ModuleNotFoundError: No module named 'numba'` | numba not installed or wrong version for Python 3.13+ | `pip install 'numba>=0.60'` |
| `_tkinter.TclError: no display name` | matplotlib trying to open a GUI window on a headless server | Use `samapy-run --no-gui`, or set environment variable `MPLBACKEND=Agg` |
| `FileNotFoundError: Eload.csv not found` | Package data not installed correctly | Run `pip install --force-reinstall samapy` |
| Optimization produces 0 kW for all components | All candidate solutions violate constraints (LPSP, RE_min, Budget) | Relax constraints: increase `LPSP_max_rate`, decrease `RE_min_rate`, increase `Budget` or `VarMax` |
| `samapy-config` freezes or no text appears | Terminal does not support questionary interactive prompts | Try a different terminal: `cmd.exe` or PowerShell on Windows; bash or zsh on Linux/macOS |
| Very slow optimization (many hours) | numba JIT compilation on first run, or `MaxIt`/`nPop` too large | First run includes JIT compilation (expected). Reduce `MaxIt` or `nPop` for testing. |
| Results folder is empty after optimization | Output directory not created or permissions issue | Check `output_directory` in YAML config. Ensure write permissions in that folder. |

---

## 14. Quick Reference: Minimum Working Example

The following is the minimum set of steps to run SAMA with a fresh pip installation using default bundled data:

**Step 1.** Install SAMAPy:

```bash
pip install samapy
```

Or from TestPyPI (during testing phase):

```bash
pip install -i https://test.pypi.org/simple/ --extra-index-url https://pypi.org/simple/ samapy
```

**Step 2.** Open a terminal in an empty working directory.

**Step 3.** Run the configuration wizard:

```bash
samapy-config
```

Follow all 18 sections. For a quick test, accept defaults in most sections.

**Step 4.** Run the optimization:

```bash
samapy-run
```

**Step 5.** View results in `sama_outputs/figs/` and `sama_outputs/data/Outputforplotting.csv`.

> **Expected Runtime:**
> - The first run always takes longer because numba compiles the EMS functions on first call.
> - After the first run, subsequent runs are significantly faster.
> - A typical run with `MaxIt=200`, `nPop=50` takes **5–30 minutes** depending on hardware and which components are enabled.

---

## 15. Package Module Structure

| Module / File | Description |
|---------------|-------------|
| **`sama/core/`** | |
| `Input_Data.py` | Main parameter class: defines all input data and default values. Singleton pattern: `InData = Input_Data()` |
| `Fitness.py` | Fitness evaluation function: calls EMS for each candidate design and returns NPC, LCOE, LEM, LPSP, and constraint penalties |
| **`sama/cli/`** | |
| `wizard.py` | Entry point for `samapy-config` command: launches `sama_Wizard.py` |
| `sama_Wizard.py` | Interactive 18-section configuration wizard using questionary |
| `runner.py` | Entry point for `samapy-run` command: loads YAML, applies config, runs selected algorithm |
| `config_loader.py` | `load_config()`, `apply_config()`, `save_config()`, `merge_configs()` functions |
| **`sama/optimizers/`** | |
| `swarm.py` | Particle Swarm Optimization (`Swarm` class) |
| `AdvancedDifferentialEvolution.py` | Advanced Differential Evolution (`AdvancedDifferentialEvolution` class) |
| `ArtificialBeeColony.py` | Improved Artificial Bee Colony (`ImprovedArtificialBeeColony` class) |
| `GreyWolfOptimizer.py` | Grey Wolf Optimizer (`GreyWolfOptimizer` and `ParallelGreyWolfOptimizer` classes) |
| `pso.py` / `ade.py` / `run_abc.py` / `gwo.py` | Thin wrapper scripts for running each algorithm directly from command line |
| **`sama/ems/`** | |
| `EMS.py` | Base hourly dispatch engine (Numba JIT compiled) |
| `EMS_EV.py` | Extended dispatch engine with EV smart charging and V2X lookahead arbitrage |
| `EMS_HP.py` | Heat pump energy attribution post-processor |
| **`sama/models/`** | |
| `Battery_Model.py` | Battery discharge and degradation models (lead-acid KiBaM and Li-ion) |
| `BB_HP_Goodman.py` | Goodman air-source heat pump black-box model |
| `BB_HP_Bosch.py` | Bosch air-source heat pump black-box model |
| **`sama/pricing/`** | |
| `Electricity_Bill_Calculator.py` | Monthly and annual billing calculations with escalation and NEM reconciliation |
| `calcFlatRate.py` | Flat rate tariff calculator |
| `calcSeasonalRate.py` | Seasonal rate tariff calculator |
| `calcMonthlyRate.py` | Monthly rate tariff calculator |
| `calcTieredRate.py` | Tiered rate tariff calculator |
| `calcSeasonalTieredRate.py` | Seasonal tiered rate calculator |
| `calcMonthlyTieredRate.py` | Monthly tiered rate calculator (3-tier) |
| `calcMonthlyTieredRate4.py` | Monthly tiered rate calculator (4-tier, for Enbridge EGD) |
| `calcTouRate.py` | Time-of-Use (TOU) rate calculator |
| `calcULTouRate.py` | Ultra-Low TOU (ULO) rate calculator |
| `service_charge.py` | Monthly utility service charge calculator (flat or tiered) |
| `Advanced_multi_cashflow.py` | Advanced multi-run cash flow analysis and comparison plots |
| **`samapy/results/`** | |
| `Results.py` | `Gen_Results()`: generates all output charts and the `Outputforplotting.csv` data file |
| **`samapy/utilities/`** | |
| `daysInMonth.py` | Calendar utility: leap-year aware days-per-month array |
| `dataextender.py` | Expand monthly averages to 8,760-hour arrays |
| `generic_load.py` | Generic electrical load profile generator (July-peak and January-peak archetypes) |
| `sam_monofacial_poa.py` | SAM NSRDB API integration: plane-of-array irradiance and temperature from METEO.csv |
| `top_down.py` | Top-down PV pricing calculator |
| `EV_Presence.py` | Determine hourly EV home-presence schedule from departure/arrival times and holidays |
| `EV_travel.py` | EV travel energy consumption modeling |
| `EV_demand_dest.py` | EV demand and destination distribution utilities |
| `Ev_Battery_Throughput.py` | EV battery lifetime throughput calculation based on degradation and km driven |
| **`samapy/content/`** | |
| *(data files)* | Bundled CSV, Excel, and meteorological data files (see Section 11) |

---

## 16. License and Citation

### 16.1 License

SAMA is released under the **GNU General Public License v3.0 (GPL-3.0)**. You are free to use, copy, modify, and distribute SAMA under the terms of this license. Any derivative works must also be released under GPL-3.0.

### 16.2 Authors

| Author | Affiliation | ORCID |
|--------|-------------|-------|
| Seyyed Ali Sadat | Free Appropriate Sustainability Technology (FAST) Research Group, Western University, London, Ontario, Canada | 0000-0001-9690-4239 |

### 16.3 Citing SAMAPy

If you use SAMA in academic work, please cite the primary published paper [A1]. A Journal of Open Source Software (JOSS) paper is currently pending submission; once published, that citation should also be included. In the meantime, please cite the GitHub repository and acknowledge the FAST Research Group at Western University.

**Primary citation (published):**

> Sadat, S.A., Takahashi, J. and Pearce, J.M., 2023. A Free and open source microgrid optimization tool: SAMA the solar alone Multi-Objective Advisor. *Energy Conversion and Management*, 298, p.117686. https://doi.org/10.1016/j.enconman.2023.117686

**JOSS paper (pending submission):**

> Sadat, S.A. and Pearce, J.M. SAMA: Solar Alone Multi-Objective Advisor. *Journal of Open Source Software (JOSS)*. Pending submission.

**Additional SAMA publications (see Section 19 for full list):**

> Sadat & Pearce (2024). Economic grid defection analysis. *Solar Energy*, 282, 112910.

> Sadat & Pearce (2025). Electricity pricing structures in Canada. *Renewable Energy*, 242, 122456.

**Repository:** https://github.com/Sas1997/SAMA | FAST Research Group, Western University | GPL-3.0 License

---

## 17. Examples

This section walks through five complete, self-contained examples covering the most common SAMAPy use cases. Each example can be run as a standalone Python script from your working directory, provided SAMAPy is installed and the relevant input files are available.

The full example scripts are also available in the repository at:
> https://github.com/Sas1997/SAMA/tree/main/SAMA%20Py%20Package/examples

---

### Example 1: Basic Grid-Tied Solar + Battery

**Scenario:** A residential customer with a simple grid-tied solar PV + Li-ion battery system. No wind, no diesel generator, no heat pump, no EV. Electricity is priced on Ontario's Ultra-Low Overnight (ULO) TOU tariff. The goal is to find the optimal number of PV panels and battery units that minimises the 25-year Net Present Cost subject to LPSP < 0.001% and minimum 50% renewable energy fraction.

```python
"""
Example 1: Basic Grid-Tied Solar + Battery
============================================
Components : PV + Li-ion battery + grid
Algorithm  : PSO (100 iterations, 30 particles)
Rate       : Ultra-Low Overnight TOU (Ontario ULO)
Output     : samapy_outputs_ex1/
"""

import numpy as np
from samapy.core.Input_Data import InData

InData.PV       = 1;  InData.WT  = 0;  InData.DG  = 0
InData.Bat      = 1;  InData.Li_ion = 1;  InData.Lead_acid = 0
InData.Grid     = 1;  InData.HP  = 0;  InData.EV  = 0;  InData.NEM = 1

InData.MaxIt = 100;  InData.nPop = 30;  InData.Run_Time = 1
InData.n = 25;  InData.year = 2023
InData.VarMin = np.array([0, 0, 0, 0, 0])
InData.VarMax = np.array([60, 0, 60, 0, 60])

InData.LPSP_max_rate = 0.000999;  InData.RE_min_rate = 50
InData.Budget = 200_000

from samapy.pricing.calcULTouRate import calcULTouRate
InData.Cbuy = calcULTouRate(
    InData.year,
    np.array([0.284, 0.284]),   # on-peak $/kWh
    np.array([0.122, 0.122]),   # mid-peak
    np.array([0.076, 0.076]),   # off-peak
    np.array([0.028, 0.028]),   # ultra-low overnight
    np.array([[16,17,18,19,20],[16,17,18,19,20]], dtype=object),
    np.array([[7,8,9,10,11,12,13,14,15,21,22],[7,8,9,10,11,12,13,14,15,21,22]], dtype=object),
    np.array([[23,0,1,2,3,4,5,6],[23,0,1,2,3,4,5,6]], dtype=object),
    np.array([0]*12), InData.daysInMonth, InData.holidays,
    treat_special_days_as_offpeak=True
)

import samapy.optimizers.swarm as swarm_mod
swarm_mod.PV = InData.PV;  swarm_mod.WT = InData.WT
swarm_mod.Bat = InData.Bat;  swarm_mod.DG = InData.DG
swarm_mod.nPop = InData.nPop;  swarm_mod.MaxIt = InData.MaxIt
swarm_mod.VarMin = InData.VarMin;  swarm_mod.VarMax = InData.VarMax

from samapy.optimizers.swarm import Swarm
Swarm().optimize()
```

**Expected output (indicative):**
```
  Example 1: Optimal System (PSO)
  ==================================================
  PV panels  : 14.2
  Battery    : 6 units
  Inverter   : 8.4 kW
  Best NPC   : $42,150
```

---

### Example 2: Full Hybrid System (PV + Wind + Battery + Diesel)

**Scenario:** An off-grid or weak-grid community energy system combining solar PV, a small wind turbine, a Li-ion battery bank, and a diesel generator as backup. No grid. Uses ADE as the optimizer.

```python
"""
Example 2: Full Hybrid System
===============================
Components : PV + WT + Battery + Diesel Generator (no grid)
Algorithm  : ADE
"""

import numpy as np
from samapy.core.Input_Data import InData

InData.PV = 1;  InData.WT = 1;  InData.DG = 1
InData.Bat = 1;  InData.Li_ion = 1;  InData.Grid = 0;  InData.NEM = 0
InData.HP = 0;  InData.EV = 0

InData.MaxIt = 200;  InData.nPop = 50;  InData.Run_Time = 1
InData.VarMin = np.array([0, 0, 0, 0, 0])
InData.VarMax = np.array([60, 20, 60, 5, 60])

import samapy.optimizers.AdvancedDifferentialEvolution as ade_mod
ade_mod.PV = InData.PV;  ade_mod.WT = InData.WT
ade_mod.Bat = InData.Bat;  ade_mod.DG = InData.DG
ade_mod.nPop = InData.nPop;  ade_mod.MaxIt = InData.MaxIt
ade_mod.VarMin = InData.VarMin;  ade_mod.VarMax = InData.VarMax

from samapy.optimizers.AdvancedDifferentialEvolution import AdvancedDifferentialEvolution
AdvancedDifferentialEvolution().optimize()
```

---

### Example 3: Heat Pump Integration

**Scenario:** A grid-tied solar PV + battery system with a Bosch air-source heat pump replacing a gas furnace. Includes both electrical and thermal load modelling.

```python
"""
Example 3: Heat Pump Integration
==================================
Components : PV + Battery + Grid + Bosch Heat Pump
Algorithm  : ADE
"""

import numpy as np
from samapy.core.Input_Data import InData

InData.PV = 1;  InData.WT = 0;  InData.DG = 0
InData.Bat = 1;  InData.Grid = 1;  InData.NEM = 1
InData.HP = 1;  InData.HP_brand = 'Bosch';  InData.EV = 0

InData.MaxIt = 200;  InData.nPop = 50;  InData.Run_Time = 1
InData.VarMin = np.array([0, 0, 0, 0, 0])
InData.VarMax = np.array([40, 0, 30, 0, 40])

from samapy.optimizers.AdvancedDifferentialEvolution import AdvancedDifferentialEvolution
AdvancedDifferentialEvolution().optimize()
```

**Heat pump figures generated:**
- `hp_ambient_conditions.png`, hourly ambient temperature and pressure conditions
- `hp_heating_performance.png`, heating power and COP over the year
- `hp_cooling_performance.png`, cooling power and COP over the year
- `hp_cop_vs_temp.png`, COP as a function of ambient temperature
- `hp_monthly_summary.png`, monthly heat pump energy and COP summary

---

### Example 4: Electric Vehicle with V2G

**Scenario:** Grid-tied PV + battery + EV with vehicle-to-grid (V2G) arbitrage. The EV charges during ultra-low overnight rate periods and discharges during peak-price hours.

```python
"""
Example 4: Electric Vehicle with V2G
=======================================
Components : PV + Battery + Grid + EV (V2G enabled)
Algorithm  : GWO
"""

import numpy as np
from samapy.core.Input_Data import InData

InData.PV = 1;  InData.WT = 0;  InData.DG = 0
InData.Bat = 1;  InData.Grid = 1;  InData.NEM = 1
InData.HP = 0;  InData.EV = 1

InData.C_ev = 82;  InData.Pev_max = 11
InData.SOCe_min = 0.03;  InData.SOCe_max = 0.97
InData.Tin = 17;  InData.Tout = 8   # arrive 5 PM, depart 8 AM
InData.Daily_trip = 68
InData.treat_special_days_as_home = False

from samapy.utilities.EV_Presence import determine_EV_presence
InData.EV_p = determine_EV_presence(
    InData.year, InData.Tout, InData.Tin,
    InData.holidays, InData.treat_special_days_as_home
)

InData.MaxIt = 150;  InData.nPop = 40
InData.VarMin = np.array([0, 0, 0, 0, 0])
InData.VarMax = np.array([40, 0, 30, 0, 40])

from samapy.optimizers.GreyWolfOptimizer import GreyWolfOptimizer
GreyWolfOptimizer().optimize()
```

**EV figures generated:**
- `EV Energy.png`, annual EV SOC and charge/discharge profile
- `EV Sp Results 1.png`, 48-hour detail (EV arbitrage against peak prices)
- `EV Sp Results 2.png`, 48-hour detail for a winter weekend

---

### Example 5: YAML-Driven Workflow (Recommended Production Workflow)

**Scenario:** Run `samapy-config` once interactively to produce a YAML, then load, optionally modify, and run the optimization programmatically. Ideal for batch parameter sweeps.

```python
"""
Example 5: YAML-Driven Workflow
==================================
Load a wizard-generated config, override specific parameters,
and launch optimization, recommended for production use.
"""

from samapy.cli.config_loader import load_config, apply_config

config = load_config("samapy_config_COMPLETE.yaml")

# Optionally override parameters before applying
config['n_ir_rate'] = 5.0          # higher discount rate sensitivity test
config['RE_incentives_rate'] = 0   # remove tax credit for comparison

indata = apply_config(config)
print(f"Annual load  : {indata.Eload.sum():.0f} kWh")
print(f"Avg Cbuy     : ${indata.Cbuy.mean():.4f}/kWh")

algo = config.get('optimization_algorithm', 'ade').lower()
if algo == 'pso':
    from samapy.optimizers.swarm import Swarm
    Swarm().optimize()
elif algo == 'ade':
    from samapy.optimizers.AdvancedDifferentialEvolution import AdvancedDifferentialEvolution
    AdvancedDifferentialEvolution().optimize()
elif algo == 'gwo':
    from samapy.optimizers.GreyWolfOptimizer import GreyWolfOptimizer
    GreyWolfOptimizer().optimize()
elif algo == 'abc':
    from samapy.optimizers.ArtificialBeeColony import ImprovedArtificialBeeColony
    ImprovedArtificialBeeColony().optimize()
```

### Batch parameter sweep

```python
from samapy.cli.config_loader import load_config, apply_config
from samapy.optimizers.swarm import Swarm
import numpy as np

base_config = load_config("samapy_config_COMPLETE.yaml")
base_config['MaxIt'] = 50;  base_config['nPop'] = 20

results = []
for incentive in [0, 10, 20, 30]:
    cfg = dict(base_config)
    cfg['RE_incentives_rate'] = incentive
    indata = apply_config(cfg)
    optimizer = Swarm()
    optimizer.optimize()
    best_idx = np.argmin(optimizer.solution_best_costs)
    X   = optimizer.solution_best_positions[best_idx]
    npc = optimizer.solution_best_costs[best_idx]
    results.append({'incentive_%': incentive, 'NPC': npc,
                    'N_PV': round(X[0],1), 'N_bat': round(X[2])})

print(f"{'Incentive %':>12} {'NPC ($)':>14} {'PV panels':>10} {'Batteries':>10}")
for r in results:
    print(f"{r['incentive_%']:>12} {r['NPC']:>14,.0f} {r['N_PV']:>10} {r['N_bat']:>10}")
```

### Tips and common pitfalls

**Using your own weather and load data:**
```python
import shutil
from samapy import get_content_path

shutil.copy("path/to/my_Eload.csv",      get_content_path("Eload.csv"))
shutil.copy("path/to/my_METEO.csv",      get_content_path("METEO.csv"))
shutil.copy("path/to/my_house_load.xlsx", get_content_path("house_load.xlsx"))
```

**Running on a headless server (no display):**
```python
import matplotlib
matplotlib.use('Agg')   # must be before any other matplotlib import
from samapy.optimizers.swarm import Swarm
Swarm().optimize()
```
Or use the CLI flag: `samapy-run --no-gui`

**Inspecting hourly results:**
```python
import pandas as pd
df = pd.read_csv("samapy_outputs/data/Outputforplotting.csv")
print(f"PV generation  : {df['Ppv'].sum():.0f} kWh")
print(f"Grid purchases : {df['Pbuy total'].sum():.0f} kWh")
print(f"Grid sales     : {df['Psell'].sum():.0f} kWh")
```

---

## 18. Sample Optimization Output

This section shows the complete terminal output and generated figures from a real SAMAPy run using the ADE optimizer with a PV + grid + Bosch heat pump + EV configuration for a London, Ontario residential property.

### 18.1 Configuration Used

- **Components:** Solar PV, Grid (ULO tariff), Bosch heat pump (48,000 BTU), EV
- **Algorithm:** ADE, MaxIt = 200, nPop = 50
- **Location:** London, Ontario (TMY 2021)
- **Project lifetime:** 25 years

### 18.2 Optimizer Convergence

The following shows the ADE convergence as printed to the terminal during a run:

```
🚀 Starting ADE optimization...
Run 0, Iteration   1, Best Cost = 14400001.689, Mean Cost = 24670212999328560.000, Success Rate = 0.000
Run 0, Iteration  10, Best Cost =   900002.731, Mean Cost =       215268420911.605, Success Rate = 0.276
Run 0, Iteration  20, Best Cost =   900001.255, Mean Cost =       109192315109.214, Success Rate = 0.256
Run 0, Iteration  30, Best Cost =   100000.767, Mean Cost =         1667996892.145, Success Rate = 0.241
Run 0, Iteration  40, Best Cost =        0.764, Mean Cost =            7407633.818, Success Rate = 0.244
Run 0, Iteration  50, Best Cost =        0.764, Mean Cost =             557501.670, Success Rate = 0.242
Run 0, Iteration  60, Best Cost =        0.764, Mean Cost =               2000.765, Success Rate = 0.226
Run 0, Iteration  70, Best Cost =        0.764, Mean Cost =                  0.764, Success Rate = 0.215
...
Run 0, Iteration 200, Best Cost =        0.764, Mean Cost =                  0.764, Success Rate = 0.090
✓ Saved: samapy_outputs/Optimization.png

✅ Optimization completed in 165.4s
📁 Results saved to: samapy_outputs/
```

The sharp drop from iteration 30 to 40 is characteristic of ADE finding the feasible region (solutions satisfying LPSP, RE, and budget constraints). Once all particles converge to feasible space, the mean cost collapses to match the best cost.

### 18.3 System Size Results

```
-------------------System Size--------------------
Cpv  (kW)       = 12.0
Cbat (kWh)      = 0
Cdg  (kW)       = 0
Cinverter (kW)  = 9.77

Selected heat pump: Bosch 1×48000 BTU  (capacity: 48,000 BTU/hr)
```

### 18.4 Economic Results

```
*****************Economic Results*****************
NPC                                  = $ 79,492.41
NPC without incentives               = $ 80,459.92
Total Solar Cost                     = $ 10,549.67
NPC for only Grid connected system   = $ 84,719.59
Total Heating Cost                   = $ 15,313.37
Total Cooling Cost                   = $  2,301.36
Total Heat Pump Cost                 = $ 17,614.72
NPC for only Natural Gas Grid        = $ 30,081.34
Total Grid avoidable cost            = $ 73,847.16
Total Grid unavoidable cost          = $ 10,872.43
Total avoided costs                  = $ 84,688.58
Total net avoided costs by HES       = $ 70,736.99
Total grid earning                   = $ 11,375.60
Total grid costs                     = $ 44,032.92
Total grid costs for the property    = $ 37,793.80
LCOE                                 = 0.16 $/kWh
LCOE (grid-only)                     = 0.25 $/kWh
Operating Cost                       = $  3,162.43
Initial Cost                         = $  7,513.51
Total replacement cost               = $ 19,198.61
Total O&M cost                       = $  8,747.37
Total incentives received            = $    967.51
The IRR of the project               = 22.18%
The Payback Period                   = 4 years
The ROI of the project               = 469.52%
```

### 18.5 Technical Results

```
================Technical Results=================
PV Power              = 11,335.6 kWh/yr
RE fraction           = 50.14%
LPSP (property)       = 0.06%
Annual Property Load  = 14,999.7 kWh
Annual EV Load        = 2,811.1 kWh
Annual Total Load     = 17,810.8 kWh
Annual Heating Load   = 17,682.4 kWh/yr
Annual Cooling Load   = 13,201.6 kWh/yr
HP Average heating COP = 4.16
HP Average cooling COP = 5.23
Annual grid purchased  = 11,018.2 kWh
Annual grid sold       = 3,853.3 kWh
```

### 18.6 Scenario Comparison: With vs Without Hybrid Energy System

The table below compares the 25-year costs with and without the hybrid energy system (HES):

| Metric | Without HES | With HES | Savings |
|--------|-------------|----------|---------|
| Year 1 Electricity Cost (PV) | $2,905.80 | $1,510.29 | $1,395.51 |
| Year 1 Natural Gas Cost (PV) | $1,031.76 | $0.00 | $1,031.76 |
| **Year 1 Total (PV)** | **$3,937.56** | **$1,510.29** | **$2,427.27 (61.6%)** |
| 25-Year Electricity (Nominal) | $93,757.94 | $48,730.60 | $45,027.34 |
| 25-Year Natural Gas (Nominal) | $33,290.58 | $0.00 | $33,290.58 |
| **25-Year Total (Nominal)** | **$127,048.51** | **$48,730.60** | **$78,317.92 (61.6%)** |
| 25-Year Total (Present Value) | $114,800.92 | $44,032.92 | $70,768.00 |

**Monthly bill comparison, Year 1 (Present Value):**

| Month | Without HES | With HES | Total Savings | % Saved |
|-------|-------------|----------|---------------|---------|
| January | $476.95 | $263.09 | $213.86 | 44.8% |
| February | $397.22 | $187.66 | $209.56 | 52.8% |
| March | $353.53 | $135.85 | $217.67 | 61.6% |
| April | $279.43 | $44.35 | $235.08 | 84.1% |
| May | $229.01 | $7.18 | $221.83 | 96.9% |
| June | $281.37 | $49.31 | $232.06 | 82.5% |
| July | $374.63 | $122.23 | $252.41 | 67.4% |
| August | $322.09 | $113.45 | $208.64 | 64.8% |
| September | $256.66 | $80.61 | $176.05 | 68.6% |
| October | $244.37 | $119.61 | $124.76 | 51.1% |
| November | $288.41 | $148.70 | $139.71 | 48.4% |
| December | $433.89 | $238.25 | $195.63 | 45.1% |
| **Total** | **$3,937.56** | **$1,510.29** | **$2,427.27** | **61.6%** |

### 18.7 Output Figures

All figures are saved to `samapy_outputs/figs/`. Below are the actual sample outputs from this example run.

---

#### Optimizer Convergence

![Optimization convergence curve](https://raw.githubusercontent.com/Sas1997/SAMA/main/Docs/SAMA%20results%20example%20outputs/Optimization.png)

*ADE convergence: best cost (blue) and mean cost (orange) vs. iteration. The sharp drop around iteration 40 marks the point where all particles enter the feasible region.*

---

#### Cash Flow

![25-year cash flow breakdown](https://raw.githubusercontent.com/Sas1997/SAMA/main/Docs/SAMA%20results%20example%20outputs/Cash_Flow.svg)

*25-year lifecycle cost breakdown: capital, replacement, O&M, grid costs, and NEM export earnings.*

---

#### Energy Balance

![Energy Balance](https://raw.githubusercontent.com/Sas1997/SAMA/main/Docs/SAMA%20results%20example%20outputs/Energy%20Balance.png)

*Hourly energy supply and demand balance across all 8,760 hours of the year.*

---

#### Energy Distribution

![Energy Distribution](https://raw.githubusercontent.com/Sas1997/SAMA/main/Docs/SAMA%20results%20example%20outputs/Energy%20Distribution.png)

*Annual energy supply by source and demand by use category.*

---

#### Electricity Cost Comparison, Annual

![Annual electricity cost comparison](https://raw.githubusercontent.com/Sas1997/SAMA/main/Docs/SAMA%20results%20example%20outputs/electricity_comparison_annual.png)

*Annual electricity cost: optimised HES vs. grid-only baseline over the 25-year project lifetime.*

---

#### Electricity Cost Comparison, Monthly

![Monthly electricity cost comparison](https://raw.githubusercontent.com/Sas1997/SAMA/main/Docs/SAMA%20results%20example%20outputs/electricity_comparison_monthly.png)

*Monthly electricity cost breakdown for Year 1: HES system vs. grid-only.*

---

#### Daily / Monthly / Yearly Average Cost of Energy System

![Average cost of energy system](https://raw.githubusercontent.com/Sas1997/SAMA/main/Docs/SAMA%20results%20example%20outputs/Daily-Monthly-Yearly%20average%20cost%20of%20energy%20system.png)

*Daily, monthly, and annual average costs of the optimised hybrid energy system.*

---

#### Daily / Monthly / Yearly Average Cost of Grid-Only System

![Average cost of grid-only system](https://raw.githubusercontent.com/Sas1997/SAMA/main/Docs/SAMA%20results%20example%20outputs/Daily-Monthly-Yearly%20average%20cost%20of%20only%20grid-connected%20system.png)

*Same timescales for the baseline grid-only scenario, for direct comparison with the HES.*

---

#### Daily / Monthly / Yearly Average Hourly Grid Cost

![Average hourly grid cost](https://raw.githubusercontent.com/Sas1997/SAMA/main/Docs/SAMA%20results%20example%20outputs/Daily-Monthly-Yearly%20average%20hourly%20cost%20of%20connecting%20to%20the%20grid.png)

*Hourly grid cost patterns broken down by season and time of day (ULO tariff structure visible).*

---

#### Daily / Monthly / Yearly Average Grid Export Earnings

![Grid export earnings](https://raw.githubusercontent.com/Sas1997/SAMA/main/Docs/SAMA%20results%20example%20outputs/Daily-Monthly-Yearly%20average%20earning%20Sell%20to%20the%20Grid.png)

*Revenue from NEM grid exports across daily, monthly, and annual timescales.*

---

#### Grid Hourly Cost

![Grid hourly cost profile](https://raw.githubusercontent.com/Sas1997/SAMA/main/Docs/SAMA%20results%20example%20outputs/Grid%20Hourly%20Cost.png)

*Hourly grid electricity cost profile across the full year, showing ULO pricing patterns.*

---

#### Heat Pump, Ambient Conditions

![Heat pump ambient conditions](https://raw.githubusercontent.com/Sas1997/SAMA/main/Docs/SAMA%20results%20example%20outputs/hp_ambient_conditions.png)

*Hourly ambient temperature and pressure at the site, the conditions driving heat pump performance. (HP = 1 only)*

---

#### Heat Pump, Heating Performance

![Heat pump heating performance](https://raw.githubusercontent.com/Sas1997/SAMA/main/Docs/SAMA%20results%20example%20outputs/hp_heating_performance.png)

*Hourly heat pump electrical power consumption and COP in heating mode over the year. (HP = 1 only)*

---

#### Heat Pump, Cooling Performance

![Heat pump cooling performance](https://raw.githubusercontent.com/Sas1997/SAMA/main/Docs/SAMA%20results%20example%20outputs/hp_cooling_performance.png)

*Hourly heat pump electrical power consumption and COP in cooling mode over the year. (HP = 1 only)*

---

#### Heat Pump, COP vs. Temperature

![Heat pump COP vs temperature](https://raw.githubusercontent.com/Sas1997/SAMA/main/Docs/SAMA%20results%20example%20outputs/hp_cop_vs_temp.png)

*Coefficient of Performance as a function of ambient temperature for both heating and cooling modes. (HP = 1 only)*

---

#### Heat Pump, Monthly Summary

![Heat pump monthly summary](https://raw.githubusercontent.com/Sas1997/SAMA/main/Docs/SAMA%20results%20example%20outputs/hp_monthly_summary.png)

*Monthly heat pump energy consumption and average COP for heating and cooling. (HP = 1 only)*

---

#### EV Energy Profile

![EV energy profile](https://raw.githubusercontent.com/Sas1997/SAMA/main/Docs/SAMA%20results%20example%20outputs/EV%20Energy.png)

*Annual EV battery SOC, charging, and discharging profile. (EV = 1 only)*

---

#### EV Detailed Results 1

![EV detailed results 1](https://raw.githubusercontent.com/Sas1997/SAMA/main/Docs/SAMA%20results%20example%20outputs/EV%20Sp%20Results%201.png)

*48-hour detailed EV power flow for a representative summer weekday showing V2G arbitrage against peak ULO prices. (EV = 1 only)*

---

#### EV Detailed Results 2

![EV detailed results 2](https://raw.githubusercontent.com/Sas1997/SAMA/main/Docs/SAMA%20results%20example%20outputs/EV%20Sp%20Results%202.png)

*48-hour detailed EV power flow for a representative winter weekend. (EV = 1 only)*

---

## 19. Technical Framework and Mathematical Formulation

This section presents the mathematical models and equations that underpin SAMA's simulation and optimization engine. The framework described here is drawn from the original SAMA technical documentation [A1] and peer-reviewed literature. All equation numbers correspond to the formulations implemented in the source code.

### 19.1 Particle Swarm Optimization Algorithm

The default optimizer in SAMA is the Particle Swarm Optimizer (PSO), a bio-inspired algorithm that moves a population of candidate solutions (particles) through the search space according to the following velocity and position update equations [A1]:

$$V_i^{t+1} = W \cdot V_i^t + c_1 U_1^t \left(P_{b_1} - P_i^t\right) + c_2 U_2^t \left(g_b^t - P_i^t\right) \quad (1)$$

$$P_i^{t+1} = P_i^t + v_i^{t+1} \quad (2)$$

In Eq. 1, $W$ is the inertia weight, $c_1$ is the cognitive (personal learning) constant, $c_2$ is the social (global learning) constant, $U_1$ and $U_2$ are random numbers drawn uniformly from $[0,1]$, $P_b$ is the particle's personal best position, and $g_b$ is the global best position of the swarm. Eq. 2 updates the particle position using the new velocity. Default PSO parameters in SAMA are $W = 1$, $c_1 = 2$, $c_2 = 2$, with an inertia weight damping ratio of 0.99 per iteration.

### 19.2 Solar PV Power Output Model

The hourly power output of PV modules ($P_{PV}$ [kW]) and corresponding PV energy ($E_{PV}$ [kWh]) are functions of module temperature ($T_{Module}$ [°C]) and plane-of-array irradiance ($POA$ [W/m²]) [13][14][15]:

$$P_{PV}\ [\text{kW}] = N_{PV} \times f_{PV} \times P_{PV}^{STC} \times \left(\frac{POA}{POA_{STC}}\right)\left[1 + \delta_{PV}\left(T_{Module} - T_{Ref}\right)\right]; \quad E_{PV} = P_{PV} \times t \quad (3)$$

Where $N_{PV}$ is the optimum number of PV modules, $f_{PV}$ is the PV derating factor (default 0.9), $P_{PV}^{STC}$ [kW] is the rated capacity at Standard Test Conditions (STC: 1000 W/m², 25 °C), $POA_{STC} = 1000$ W/m², $\delta_{PV}$ is the temperature coefficient at STC (default $-3.7 \times 10^{-3}$ /°C), and $T_{Ref} = 25$ °C.

The module operating temperature $T_{Module}$ is calculated using the Nominal Operating Cell Temperature (NOCT) model [17]:

$$T_{Module}\ [°C] = T_{Amb} + \left(\frac{T_{noct} - 20}{800}\right) \times POA \quad (4)$$

Where $T_{noct}$ [°C] is the nominal operating cell temperature (default 45 °C) and $T_{Amb}$ [°C] is the ambient temperature from the meteorological dataset.

### 19.3 Battery Storage Model (KiBaM)

SAMA uses the Kinetic Battery Model (KiBaM) [18] to model energy storage. KiBaM represents the battery as two reservoirs (available and bound energy). The maximum chargeable power $P_{BT,ch}^{max,KiBaM}$ [kW] is given by [18]:

$$P_{BT,ch}^{max,KiBaM} = \frac{kcQ_{Max} + kQ_1 e^{-kt} + Qkc\left(1 - e^{-kt}\right)}{1 - e^{-kt} + c(k\Delta t - 1 + e^{-kt})}; \quad E_{BT,ch}^{max,KiBaM} = P_{BT,ch}^{max,KiBaM} \times t \quad (5)$$

Following HOMER Pro [23], SAMA also applies two additional charging limits based on the maximum charge rate ($\alpha$ [A/Ah]) and maximum current ($I_{max}$):

$$P_{BT,ch}^{max,mcr} = \frac{(1 - e^{-\alpha t})(Q_{max} - Q)}{t}; \quad E_{BT,ch}^{max,mcr} = P_{BT,ch}^{max,mcr} \times t \quad (6)$$

$$P_{BT,ch}^{max,mcc} = \frac{N_{BT} I_{max} V_{nom}}{1000}; \quad E_{BT,ch}^{max,mcc} = P_{BT,ch}^{max,mcc} \times t \quad (7)$$

The actual maximum charge power is the minimum of the three limits divided by round-trip efficiency:

$$P_{BT,ch}^{max} = \frac{\min\left(P_{BT,ch}^{max,KiBaM},\ P_{BT,ch}^{max,mcr},\ P_{BT,ch}^{max,mcc}\right)}{\eta_{BT}}; \quad E_{BT,ch}^{max} = P_{BT,ch}^{max} \times t \quad (8)$$

The battery wear cost ($Cost_{BT}^{Wear}$ [$/kWh]) represents the marginal cost of each kWh cycled through the battery [24][25][26]:

$$Cost_{BT}^{Wear} = \frac{R_{BT} \times N_{BT}^{total}}{N_{bat} \times Q_{lifetime} \times \sqrt{\eta_{BT}}} \quad (9)$$

Battery energy for the next time-step is updated as [A1]:

$$E_{BT}(t+1) = (1 - \delta) \times E_{BT}(t) + \eta_{BT} \times E_{BT}^{ch}(t) - \frac{E_{BT}^{dch}(t)}{\eta_{BT}} \quad (10)$$

### 19.4 Backup (Diesel or Gasoline) Generator Model

The total hourly operating cost of the DG is calculated from its fuel consumption using a linear fuel curve [26][27][28]:

$$Cost_{DG} = b \times P_{DG}^{Nominal} \times Cost_{fuel} + \frac{R_{DG} \times P_{DG}^{Nominal}}{TL_{DG}} + MO_{DG} + a \times Cost_{fuel} \quad (11)$$

Where $a$ [L/hr/kW output] and $b$ [L/hr/kW rated] are the slope and intercept of the fuel consumption curve (defaults $a = 0.2730$, $b = 0.0330$).

### 19.5 Wind Turbine Model

Hub-height wind speed is extrapolated from anemometer height $h_0$ to hub height $h_{hub}$ using the power-law wind shear model:

$$V_{hub} = V_{anemometer} \times \left(\frac{h_{hub}}{h_0}\right)^{\alpha_{wind\_turbine}} \quad (12)$$

Where $\alpha_{wind\_turbine}$ is the friction coefficient (default 0.14 for open terrain). Wind turbine output follows a piecewise power curve: zero below cut-in speed ($v_{cut\_in} = 2.5$ m/s), linear ramp from cut-in to rated speed ($v_{rated} = 9.5$ m/s), constant rated power from rated to cut-out speed ($v_{cut\_out} = 25$ m/s), and zero above cut-out.

### 19.6 Energy Management Strategy (EMS)

SAMA implements an advanced load-following dispatch strategy [29][30] that operates over all 8,760 hours of the year. When renewable generation ($E_{RE}$) exceeds load demand ($E_{load}$):

$$E_{BT}^{ch}(t) = \min\left(E_{BT}^{empty},\ E_{RE}(t) - \frac{E_{load}(t)}{\eta_{inv}}\right) \quad (13)$$

$$E_{BT}^{empty} = \frac{E_{BT}^{max} - E_{BT}}{\eta_{BT}} \quad (14)$$

$$E_{AC}^{sur} = \eta_{inv} \times \left(E_{RE}(t) - E_{BT}^{ch}(t)\right) - E_{load}(t) \quad (15)$$

When load exceeds renewable generation, SAMA selects the dispatch order among battery, DG, and grid based on marginal cost comparison. Six prioritization scenarios are defined based on the relative values of the grid buy price ($C_{buy}$), DG cost ($Cost_{DG}$), and battery wear cost ($Cost_{BT}^{Wear}$):

| Scenario | Order | Condition |
|:--------:|-------|-----------|
| 1 | Grid → DG → Battery | $C_{buy} \leq Cost_{DG}$ and $Cost_{DG} \leq Cost_{BT}^{Wear}$ |
| 2 | Grid → Battery → DG | $C_{buy} \leq Cost_{BT}^{Wear}$ and $Cost_{BT}^{Wear} < Cost_{DG}$ |
| 3 | DG → Grid → Battery | $Cost_{DG} < C_{buy}$ and $C_{buy} \leq Cost_{BT}^{Wear}$ |
| 4 | DG → Battery → Grid | $Cost_{DG} < Cost_{BT}^{Wear}$ and $Cost_{BT}^{Wear} < C_{buy}$ |
| 5 | Battery → DG → Grid | $Cost_{BT}^{Wear} < Cost_{DG}$ and $Cost_{DG} < C_{buy}$ |
| 6 | Battery → Grid → DG | All remaining cases |

The loss of power supply probability (LPSP) constraint tracks unmet energy across the year:

$$LPSP = \frac{\sum_{t}^{8760} Ens(t)}{\sum_{t}^{8760} E_{load}(t)} \quad (16)$$

The renewable fraction (RF) constraint:

$$RF = 1 - \frac{\sum_{t}^{8760} E_{non\ renewable}(t)}{\sum_{t}^{8760} \left(E_{load}(t) - ENS(t)\right)} \quad (17)$$

### 19.7 Economic Model

The real discount rate $i$ is derived from the nominal discount rate $i'$ and expected inflation rate $f$ [A1]:

$$i = \frac{i' - f}{1 + f} \quad (18)$$

The Net Present Cost (NPC) [31]:

$$NPC = C_I + \frac{C_R + C_{MO} + C_F - C_S + C_G}{(1+i)^n} \quad (19)$$

The Capital Recovery Factor (CRF) [33]:

$$CRF(i,N) = \frac{i(1+i)^n}{(1+i)^n - 1} \quad (20)$$

The Levelized Cost of Energy (LCOE) [33]:

$$LCOE = \frac{CRF \times NPC}{\sum_{0}^{8760}(E_{load} - Ens + P_{sell})} \quad (21)$$

When `EM = 1`, SAMA also minimizes Levelized Emission (LE) [34]:

$$LE = \frac{\sum_{0}^{8760} Non\ Grid\ Emissions(t) + \sum_{0}^{8760} Grid_{Emissions}(t)}{\sum_{0}^{8760}\left(E_{load}(t) - Ens(t)\right)} \quad (22)$$

The multi-objective function combines NPC and LE, with penalties applied when constraints are violated [A1]:

$$Z = NPC + EM \times LE + penalties \quad (23)$$

**Table 2.** Penalty conditions in SAMA optimization (from [A1]):

| Penalty Condition | Reason |
|-------------------|--------|
| DC to AC ratio > 1.99 × (P_Inv + P_DG + P_buy_max) | DC to AC ratio must not exceed 2 for a feasible inverter design |
| LPSP > LPSP_max | System does not supply sufficient power to meet reliability constraint |
| RF < RE_min | Renewable fraction is below the user-specified minimum |
| I_Cost > Budget | Initial capital cost exceeds the user-specified budget limit |

### 19.8 Validation Against HOMER Pro

SAMA results have been cross-validated against HOMER Pro across two geographically distinct U.S. locations: Sacramento, California and New Bern, North Carolina [A1]. Validation used identical input data (same load profiles from OpenEI [2], meteorological data from NSRDB [1], and equivalent pricing). The PSO optimizer was run with `MaxIt = 101` and `nPop = 100`, repeated 10 times to assess convergence.

Convergence was achieved by iteration 70 on the first run and as early as iteration 20 on subsequent runs. Final system configurations, LCOE, NPC, energy distribution, and battery state-of-charge profiles from SAMA showed close agreement with HOMER Pro across all tested system types (PV+Grid, PV+BT+Grid, PV+BT+DG+Grid, PV+BT+DG, PV+BT).

For the off-grid PV+DG+Battery configuration in Sacramento, the optimizer found that maintaining a grid connection (LCOE = $0.0838/kWh) is significantly more economical than full grid defection ($0.3674/kWh). Similar conclusions apply to New Bern. This capability makes SAMA particularly useful for grid-defection analysis [A2].

---

## 20. References

[1] National Solar Radiation Database (NSRDB). "Solar Resource Maps and Data." https://nsrdb.nrel.gov (accessed Nov. 07, 2022).

[2] Open Energy Data Initiative (OEDI), OpenEI. https://data.openei.org/ (accessed Apr. 14, 2023).

[3] V. Ramasamy et al., "US Solar Photovoltaic System and Energy Storage Cost Benchmarks, With Minimum Sustainable Price Analysis: Q1 2022," National Renewable Energy Lab. (NREL), Golden, CO (United States), 2022.

[4] Westinghouse. "iGen2200 Inverter Generator." https://westinghouseoutdoorpower.com (accessed Jan. 12, 2023).

[5] A.L. Bukar, C.W. Tan, L.K. Yiew, R. Ayop, and W.-S. Tan, "A rule-based energy management scheme for long-term optimal capacity planning of grid-independent microgrid optimized by multi-objective grasshopper optimization algorithm," *Energy Convers. Manag.*, vol. 221, p. 113161, Oct. 2020.

[6] J. Hirvonen and K. Siren, "A novel fully electrified solar heating system with a high renewable fraction - Optimal designs for a high latitude community," *Renew. Energy*, vol. 127, pp. 298-309, Nov. 2018.

[7] C.P. Cameron, W.E. Boyson, and D.M. Riley, "Comparison of PV system performance-model predictions with measured PV system performance," in *2008 33rd IEEE PVSC*, May 2008, pp. 1-6.

[8] G. Blaesser and E. Rossi, "Extrapolation of outdoor measurements of PV array I-V characteristics to standard test conditions," *Sol. Cells*, vol. 25, no. 2, pp. 91-96, Nov. 1988.

[9] V. Sun, A. Asanakham, T. Deethayat, and T. Kiatsiriroat, "Evaluation of nominal operating cell temperature (NOCT) of glazed photovoltaic thermal module," *Case Stud. Therm. Eng.*, vol. 28, p. 101361, Dec. 2021.

[10] L. Dunn, M. Gostein, and K. Emery, "Comparison of pyranometers vs. PV reference cells for evaluation of PV array performance," in *2012 38th IEEE PVSC*, Jun. 2012, pp. 002899-002904.

[11] M.R. Elkadeem et al., "Feasibility analysis and optimization of an energy-water-heat nexus supplied by an autonomous hybrid renewable power generation system," *Desalination*, vol. 504, p. 114952, May 2021.

[12] A.F. Guven, N. Yorukeren, and M.M. Samy, "Design optimization of a stand-alone green energy system of university campus based on Jaya-Harmony Search and Ant Colony Optimization algorithms approaches," *Energy*, vol. 253, p. 124089, Aug. 2022.

[13] A. Seifi et al., "An optimal programming among renewable energy resources and storage devices for responsive load integration," *J. Energy Storage*, vol. 27, p. 101126, Feb. 2020.

[14] S.A. Sadat, B. Hoex, and J.M. Pearce, "A Review of the Effects of Haze on Solar Photovoltaic Performance," *Renew. Sustain. Energy Rev.*, vol. 167, p. 112796, Oct. 2022. https://doi.org/10.1016/j.rser.2022.112796

[15] A.H. Mondal and M. Denich, "Hybrid systems for decentralized power generation in Bangladesh," *Energy Sustain. Dev.*, vol. 14, no. 1, pp. 48-55, Mar. 2010.

[16] A. Maleki and F. Pourfayaz, "Optimal sizing of autonomous hybrid photovoltaic/wind/battery power system with LPSP technology," *Sol. Energy*, vol. 115, pp. 471-483, May 2015.

[17] H. Lan et al., "Optimal sizing of hybrid PV/diesel/battery in ship power system," *Appl. Energy*, vol. 158, pp. 26-34, Nov. 2015.

[18] J.F. Manwell and J.G. McGowan, "Lead acid battery storage model for hybrid energy systems," *Sol. Energy*, vol. 50, no. 5, pp. 399-405, May 1993.

[19] G.P. Fenner et al., "Comprehensive Model for Real Battery Simulation Responsive to Variable Load," *Energies*, vol. 14, no. 11, 2021.

[20] L.M. Rodrigues et al., "A Temperature-Dependent Battery Model for Wireless Sensor Networks," *Sensors*, vol. 17, no. 2, 2017.

[21] Y. Zhao et al., "An iterative learning approach to identify fractional order KiBaM model," *IEEE/CAA J. Autom. Sin.*, vol. 4, no. 2, pp. 322-331, 2017.

[22] L.M. Rodrigues et al., "An analytical model to estimate the state of charge and lifetime for batteries with energy harvesting capabilities," *Int. J. Energy Res.*, vol. 44, no. 7, pp. 5243-5258, 2020.

[23] HOMER Energy. "How HOMER Calculates the Maximum Battery Charge Power." https://www.homerenergy.com/products/pro/docs/latest/ (accessed Nov. 07, 2022).

[24] F.H. Jufri et al., "Optimal Battery Energy Storage Dispatch Strategy for Small-Scale Isolated Hybrid Renewable Energy System," *Energies*, vol. 14, no. 11, 2021.

[25] M. Ashari, C.V. Nayar, and W.W.L. Keerthipala, "Optimum operation strategy and economic analysis of a photovoltaic-diesel-battery-mains hybrid UPS," *Renew. Energy*, vol. 22, no. 1, pp. 247-254, Jan. 2001.

[26] T. Lambert, P. Gilman, and P. Lilienthal, "Micropower system modeling with HOMER," *Integr. Altern. Sources Energy*, vol. 1, no. 1, pp. 379-385, 2006.

[27] H. Suryoatmojo, A.A. Elbaset, and T. Hiyama, "Economic and reliability evaluation of wind-Diesel-Battery system for isolated island considering CO2 emission," *IEEJ Trans. Power Energy*, vol. 129, no. 8, pp. 1000-1008, 2009.

[28] A. Yahiaoui et al., "Grey wolf optimizer for optimal design of hybrid renewable energy system PV-Diesel Generator-Battery," *Sol. Energy*, vol. 158, pp. 941-951, Dec. 2017.

[29] C.D. Barley, "Modeling and optimization of dispatch strategies for remote hybrid power systems," Ph.D. dissertation, Colorado State University, 1996.

[30] C.D. Barley and C.B. Winn, "Optimal dispatch strategy in remote hybrid power systems," *Sol. Energy*, vol. 58, no. 4, pp. 165-179, Oct. 1996.

[31] H. Song et al., "A novel hybrid energy system for hydrogen production and storage in a depleted oil reservoir," *Int. J. Hydrog. Energy*, vol. 46, no. 34, pp. 18020-18031, May 2021.

[32] HOMER Energy. "Salvage Value." https://www.homerenergy.com/products/pro/docs/3.11/salvage_value.html (accessed Jun. 05, 2023).

[33] D.P. Clarke, Y.M. Al-Abdeli, and G. Kothapalli, "Multi-objective optimisation of renewable hybrid energy systems with desalination," *Energy*, vol. 88, pp. 457-468, Aug. 2015.

[34] A. Parlikar et al., "The carbon footprint of island grids with lithium-ion battery systems: An analysis based on levelized emissions of energy supply," *Renew. Sustain. Energy Rev.*, vol. 149, p. 111353, Oct. 2021.

---

## 19. SAMA Publications and Further Reading

The following peer-reviewed publications describe SAMA's development, validation, and application across a range of techno-economic energy studies. Users who wish to cite SAMA in academic work should reference these papers.

[A1] Sadat, S.A., Takahashi, J. and Pearce, J.M., 2023. A Free and open-source microgrid optimization tool: SAMA the solar alone Multi-Objective Advisor. *Energy Conversion and Management*, 298, p.117686. https://doi.org/10.1016/j.enconman.2023.117686

[A2] Sadat, S.A. and Pearce, J.M., 2024. The threat of economic grid defection in the US with solar photovoltaic, battery and generator hybrid systems. *Solar Energy*, 282, p.112910. https://doi.org/10.1016/j.solener.2024.112910

[A3] Sadat, S.A. and Pearce, J.M., 2025. Techno-economic evaluation of electricity pricing structures on photovoltaic and photovoltaic-battery hybrid systems in Canada. *Renewable Energy*, 242, p.122456. https://doi.org/10.1016/j.renene.2025.122456

[A4] Sadat, S.A., Mittal, K. and Pearce, J.M., 2025. Using investments in solar photovoltaics as inflation hedges. *Energies*, 18(4), p.890. https://doi.org/10.3390/en18040890

[A5] Groza, J.M., Sadat, S.A., Hayibo, K.S. and Pearce, J.M., 2024. Using a ledger to facilitate autonomous peer-to-peer virtual net metering of solar photovoltaic distributed generation. *Solar Energy Advances*, 4, p.100064. https://doi.org/10.1016/j.seja.2024.100064

[A6] Sadat, S.A., Lemieux, J.E.B., and Pearce, J.M., 2026. Shifting Subsidies: Implications of Redirecting Alberta's Oil and Gas Government Support to Solar Power. *Energies*, 19(4), 972. https://doi.org/10.3390/en19040972

[A7] Sadat, S.A., Hoex, B. and Pearce, J.M., 2022. A review of the effects of haze on solar photovoltaic performance. *Renewable and Sustainable Energy Reviews*, 167, p.112796. https://doi.org/10.1016/j.rser.2022.112796

For the most up-to-date list of SAMA publications and to request preprints or reprints, visit the FAST Research Group GitHub at https://github.com/Sas1997/SAMA or contact the corresponding author directly.

---

## Appendix A: Where to Find Input Data for SAMA

This appendix provides practical guidance on where to obtain real-world data for each major input category in SAMA.

### A.1 Financial and Economic Parameters

**Nominal Discount Rate:** Check the Federal Reserve primary credit rate at [federalreserve.gov](https://federalreserve.gov) or financial data sites such as YCharts for U.S. projects. For non-U.S. projects, use your local central bank rate. Typical values are 2 to 5%. Input as a percentage (e.g., `4` for 4%).

**Expected Inflation Rate:** For U.S. projects, refer to the Bureau of Labor Statistics CPI reports at [bls.gov](https://bls.gov), or check Federal Reserve projections at [federalreserve.gov](https://federalreserve.gov). For other countries, use national statistics agencies. Input as a percentage.

**RE Incentives / Federal Tax Credit:** In the United States, the Investment Tax Credit (ITC) provides a 30% credit for solar under the Inflation Reduction Act. Check the Database of State Incentives for Renewables and Efficiency (DSIRE) at [dsireusa.org](https://dsireusa.org) for current federal and state programs.

### A.2 System Constraints

**LPSP:** Set to 0% for critical loads (medical, emergency). For non-critical off-grid applications, 1 to 5% is typical.

**Minimum Renewable Energy Fraction:** Align with local Renewable Portfolio Standards (RPS) or your project sustainability target. DSIRE at [dsireusa.org](https://dsireusa.org) lists U.S. state RPS requirements.

**Net Metering Cap (kW):** Check your utility net metering policy at [dsireusa.org](https://dsireusa.org). Common U.S. residential caps range from 10 to 100 kW.

**Rooftop Area Limit:** Measure usable roof area using Google Earth, satellite imagery, or a professional site survey. Subtract areas for chimneys, vents, setbacks (typically 3 feet from roof edges), and shading. As of 2025, high-efficiency monocrystalline panels require approximately 7 to 9 m² per kW installed including spacing.

### A.3 Weather and Meteorological Data (`METEO.csv`)

The bundled `METEO.csv` is for London, Ontario. For any other location, download a new file containing GHI, DNI, DHI, Temperature, Wind Speed, and Pressure columns with two header rows followed by 8,760 hourly rows.

- **Primary source:** National Solar Radiation Database (NSRDB) at [nsrdb.nrel.gov](https://nsrdb.nrel.gov). Provides free hourly weather data for the U.S., Canada, and many other countries. Select the SAM/CSV export format.
- **Alternative:** NASA POWER at [power.larc.nasa.gov](https://power.larc.nasa.gov) provides global coverage for any coordinates.
- **POA irradiance** (for `G_type = 2`): Use PVWatts at [pvwatts.nrel.gov](https://pvwatts.nrel.gov) or SAM at [sam.nrel.gov](https://sam.nrel.gov).

> **Tip:** When downloading from NSRDB, select the **Typical Meteorological Year (TMY)** dataset rather than a single calendar year. TMY data is specifically designed for long-term energy system analysis.

### A.4 Electrical Load Data (`Eload.csv`)

`Eload.csv` requires one column of 8,760 hourly load values in kW with no header. Options to replace the bundled file:

- **Smart meter data:** Most utilities provide 15-minute or hourly interval data through their customer portal (Green Button data in North America).
- **From bills:** Use SAMA load modes 3–8 to let SAMA generate a synthetic 8,760-hour profile from your monthly kWh totals.
- **Open databases:** OpenEI at [openei.org](https://openei.org) provides representative commercial and residential load profiles by building type and U.S. climate zone.

### A.5 Thermal Load Data (`house_load.xlsx`) for Heat Pump

`house_load.xlsx` must have hourly heating load (`Hload`) in column 2 and cooling load (`Cload`) in column 3, 8,760 rows each, in kW. Options:

- **EnergyPlus:** Free DOE building simulation software at [energyplus.net](https://energyplus.net). Build a model with local weather and export hourly heating/cooling loads. Most accurate approach.
- **RETScreen:** Free software from Natural Resources Canada at [nrcan.gc.ca](https://nrcan.gc.ca).
- **Manual entry:** In `samapy-config`, set `Tload_type` to 2 or higher and enter monthly average loads.

### A.6 Solar PV Parameters

- **Module efficiency and temperature coefficient:** From the manufacturer datasheet. Use EnergySage ([energysage.com](https://energysage.com)) or the California Energy Commission (CEC) equipment list. Leading 2025 monocrystalline modules range from 20 to 25% efficiency.
- **Capital and O&M costs:** NREL cost benchmark reports at [nrel.gov](https://nrel.gov). NREL reports residential PV system costs of USD 2,500 to 3,500 per kW installed and O&M of USD 20 to 40 per kW per year for 2025.
- **Azimuth and tilt:** For the Northern Hemisphere, south-facing (180°) at a tilt equal to your site latitude maximizes annual yield. Use PVWatts ([pvwatts.nrel.gov](https://pvwatts.nrel.gov)) to optimize for your specific location.

### A.7 Wind Turbine Parameters

- **Wind speed data:** Included in the `METEO.csv` from NSRDB.
- **Turbine specs:** From the manufacturer datasheet. Typical small turbine values: cut-in 2.5–3.5 m/s, rated speed 9–13 m/s, cut-out 25 m/s.
- **Friction coefficient:** Use 0.14 for open flat terrain, 0.20 for suburban, 0.30 for forested or urban.
- **Capital costs:** NREL Distributed Wind Market Report at [nrel.gov/wind](https://nrel.gov/wind). Small residential turbines cost approximately USD 4,000 to 8,000 per kW installed in 2025.

### A.8 Battery Storage Parameters

- **Capital cost:** NREL Annual Technology Baseline at [nrel.gov](https://nrel.gov). Li-ion installed costs average USD 200 to 400 per kWh in the U.S. for 2025.
- **Round-trip efficiency and SOC limits:** From manufacturer datasheet. Li-ion: 90–95% round-trip efficiency. Lead-acid: 70–85%.
- **Lifetime throughput:** From manufacturer cycle life tables. NREL and PNNL publish battery degradation models for Li-ion.

### A.9 Diesel Generator Parameters

- **Fuel cost per liter:** U.S. EIA weekly petroleum reports at [eia.gov/petroleum](https://eia.gov/petroleum), GasBuddy at [gasbuddy.com](https://gasbuddy.com), or GlobalPetrolPrices at [globalpetrolprices.com](https://globalpetrolprices.com).
- **Fuel curve coefficients (a and b):** From the manufacturer fuel consumption data sheet. Typical: $a$ = 0.24–0.44 L/hr per kW output, $b$ = 0.01–0.11 L/hr per kW rated.
- **CO₂ emissions:** U.S. EIA default is 2.68 kg CO₂ per liter of diesel.

### A.10 Heat Pump Parameters

SAMA includes built-in performance models for Bosch and Goodman brand heat pumps. Select the brand that most closely matches your planned installation.

- **Rated capacity:** Determined by Manual J heating/cooling load calculations per ACCA standards. The required capacity is typically 15 to 30 BTU/hr per square foot depending on climate.
- **Capital cost:** EnergySage, Angi, and Home Depot provide current quotes. For 2025, mini-split systems cost approximately USD 3,000 to 8,000 installed for a 12,000 BTU unit. Federal IRA rebates (up to USD 2,000) and state incentives at [dsireusa.org](https://dsireusa.org) can reduce effective cost.

### A.11 Electric Vehicle Parameters

- **Battery capacity, range, and charging specs:** [EV-Database.org](https://ev-database.org), [InsideEVs](https://insideevs.com), or the U.S. EPA at [fueleconomy.gov](https://fueleconomy.gov).
- **Battery replacement cost:** Recurrent Auto ([recurrentauto.com](https://recurrentauto.com)) tracks real-world EV battery health and replacement cost estimates. As of 2025, battery costs are approximately USD 100 to 150 per kWh.
- **Daily trip distance:** From your own odometer records or U.S. DOT National Household Travel Survey (NHTS) at [nhts.ornl.gov](https://nhts.ornl.gov). U.S. average daily vehicle travel is approximately 50 to 65 km.

### A.12 Grid and Utility Rate Parameters

> **Grid rate parameters have the largest impact on SAMA economic results. Always use your actual utility tariff rather than national averages.**

- **Electricity rate schedule:** Download from your utility website or OpenEI Utility Rate Database at [openei.org/apps/USURDB](https://openei.org/apps/USURDB).
- **Current electricity prices:** From your most recent utility bill. U.S. EIA publishes monthly state average prices at [eia.gov/electricity](https://eia.gov/electricity).
- **Electricity price escalation:** Review historical rate filings on your state Public Utility Commission (PUC) website. U.S. average is 2 to 3% per year historically.
- **Net metering policy:** DSIRE at [dsireusa.org](https://dsireusa.org) for state rules.
- **Natural gas rates:** From your gas utility bill or U.S. EIA natural gas reports at [eia.gov/naturalgas](https://eia.gov/naturalgas).

> **Tip:** For TOU or ULO rate structures, contact your utility customer service or visit their website for the exact on-peak, mid-peak, and off-peak hour windows. These vary significantly by utility and season.

### A.13 Summary of Recommended Data Sources

| Input Category | Recommended Data Source |
|----------------|------------------------|
| Weather (`METEO.csv`) | NSRDB ([nsrdb.nrel.gov](https://nsrdb.nrel.gov)), NASA POWER ([power.larc.nasa.gov](https://power.larc.nasa.gov)) |
| POA Irradiance (`Irradiance.csv`) | PVWatts ([pvwatts.nrel.gov](https://pvwatts.nrel.gov)), SAM ([sam.nrel.gov](https://sam.nrel.gov)) |
| Electrical load (`Eload.csv`) | Utility smart meter / Green Button data, OpenEI load profiles |
| Thermal load (`house_load.xlsx`) | EnergyPlus ([energyplus.net](https://energyplus.net)), RETScreen ([nrcan.gc.ca](https://nrcan.gc.ca)) |
| PV module specs and costs | Manufacturer datasheet, EnergySage, NREL cost benchmarks |
| PV incentives | DSIRE ([dsireusa.org](https://dsireusa.org)), IRS ITC documentation |
| Wind turbine specs and costs | Manufacturer datasheet, NREL Distributed Wind Market Report |
| Battery costs | NREL Annual Technology Baseline, SEIA reports, EnergySage |
| Battery specs | Manufacturer datasheet, Battery University ([batteryuniversity.com](https://batteryuniversity.com)) |
| Diesel fuel cost | EIA weekly petroleum ([eia.gov](https://eia.gov)), GasBuddy, GlobalPetrolPrices |
| Diesel emissions factors | U.S. EPA emission factors, The Climate Registry ([theclimateregistry.org](https://theclimateregistry.org)) |
| Heat pump costs | EnergySage, Angi, DSIRE for IRA rebates |
| EV specs and battery cost | [EV-Database.org](https://ev-database.org), InsideEVs, [fueleconomy.gov](https://fueleconomy.gov), Recurrent Auto |
| Electricity rates | Your utility bill, OpenEI USURDB ([openei.org](https://openei.org)), EIA |
| Net metering policy | DSIRE ([dsireusa.org](https://dsireusa.org)), utility interconnection agreement |
| Discount rate | Federal Reserve ([federalreserve.gov](https://federalreserve.gov)), local central bank |
| Inflation rate | BLS CPI ([bls.gov](https://bls.gov)), Federal Reserve projections |
| Natural gas rates | Gas utility bill, EIA natural gas ([eia.gov/naturalgas](https://eia.gov/naturalgas)) |

---

*SAMAPy is developed and maintained by the Free Appropriate Sustainability Technology (FAST) Research Group at Western University, London, Ontario, Canada.*

*Repository: https://github.com/Sas1997/SAMA | PyPI: https://pypi.org/project/samapy | License: GPL-3.0*
