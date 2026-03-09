---
title: 'SAMA: Open-Source Multi-Objective Optimization for Hybrid Renewable Energy Microgrids with Advanced Energy Management'
tags:
  - Python
  - renewable energy
  - optimization
  - microgrid
  - solar PV
  - metaheuristic algorithms
  - energy management
  - battery storage
  - electric vehicles
  - heat pumps
authors:
  - name: Seyyed Ali Sadat
    affiliation: 1
    orcid: 0000-0001-9690-4239
  - name: Joshua M. Pearce
    affiliation: 1
affiliations:
  - name: Free Appropriate Sustainability Technology (FAST) research group, Western University, London, Ontario, Canada
    index: 1
date: 12 January 2026
bibliography: paper.bib

---

# Summary

SAMA (Solar Alone Multi-objective Advisor) is a comprehensive open-source platform for the design, sizing, and operation optimization of hybrid renewable energy microgrids. The software performs location-specific techno-economic assessment using site-dependent electrical and thermal load profiles, meteorological data, component degradation models, and utility-specific billing structures. Unlike widely used proprietary tools (HOMER Pro, HOMER Grid), which are now closed-source and license-restricted, SAMA provides transparent optimization algorithms, fully extensible architecture, and no cost barriers for researchers, practitioners, and resource-constrained institutions.

SAMA integrates multi-objective optimization via advanced metaheuristic algorithms (Differential Evolution, Artificial Bee Colony, Grey Wolf Optimizer, Particle Swarm Optimization), realistic component models for solar PV, wind turbines, battery storage (lead-acid and lithium-ion), diesel generators, air-source heat pumps, and electric vehicles, and sophisticated hourly energy management strategies that dispatch power resources under real-world utility tariff structures (time-of-use, seasonal, tiered, ultra-low pricing). The software rigorously models complex tariff calculations, net metering credits, and multi-vector energy flows while accounting for battery and EV degradation, thermal load coupling through heat pumps, and vehicle-to-everything (V2X) capabilities with lookahead-based arbitrage optimization.

Through validation studies against HOMER Pro in diverse climatic zones (Sacramento, California; New Bern, North Carolina), SAMA achieves close agreement in optimal system configurations and performance metrics while exposing trade-offs and handling constraint scenarios (rooftop area limits, net-metering caps, renewable fraction requirements, loss-of-power-supply probability) that proprietary tools handle less flexibly. 

# Statement of Need

Distributed hybrid renewable energy systems are critical for decarbonization, climate resilience, energy security in remote communities, and decentralized electrification. Design and sizing decisions are complex and require rigorous optimization across competing objectives (economics, reliability, emissions, and constraints imposed by utility regulations, geographic limitations, and emerging technologies like EVs and heat pumps).

Proprietary tools such as HOMER Pro and HOMER Grid have dominated the market but suffer from significant limitations. Originally developed as open-source research software, these tools are now closed, restricting user-developers from adapting and improving algorithms, tariff models, and component representations. Practical and research barriers include: (i) high licensing costs prohibiting use in low-resource labs, developing regions, and by independent practitioners; (ii) restricted to single- or limited-objective optimization, preventing explicit exploration of Pareto frontiers; (iii) rigid pricing methodologies that struggle to represent complex real-world tariffs (particularly time-of-use and tiered schemes); (iv) limited support for emerging technologies such as electric vehicles and heat pumps in unified frameworks; (v) no transparency into algorithms, making reproducibility and validation difficult; and (vi) inability to extend components and control strategies without vendor support.

# Software Description

## Architecture Overview

SAMA is organized as a modular, layered Python codebase consisting of input handling, component and energy system modeling, real-time dispatch optimization, financial and tariff analysis, metaheuristic optimizers, and result reporting.

### Core Modules

**Input Configuration (`Input_Data.py`)**: Comprehensive parameter specification for system design, optimization settings, component economics, utility tariff structures, and load/weather data sourcing. Supports 10+ electrical load input modes (hourly CSV, monthly averages, generic synthetic profiles), thermal load via Excel, weather data integration via SAM's National Solar Radiation Database (NSRDB) API, and on-the-fly calculation of PV and wind resources.

**Component Models and Physics**: 
- **PV (`Results.py`, `Fitness.py`)**: Hourly power output from plane-of-array irradiance using temperature-dependent efficiency model with NOCT-based cell temperature correction and soiling losses.
- **Wind (`Fitness.py`)**: Hub-height wind speed adjustment via power-law profile; piecewise power curve with cut-in, rated, and cut-out speeds.
- **Battery Storage (`Battery_Model.py`)**: Support for lead-acid (KiBaM model with charge/discharge efficiency, depth-of-discharge constraints, current limits) and lithium-ion (idealized model with fixed efficiency, voltage limits, lifetime throughput degradation).
- **Diesel Generator (`Fitness.py`, `EMS.py`)**: Fuel consumption modeled as affine function of output; minimum loading ratio constraints; lifespan and replacement scheduling.
- **Heat Pump (`BB_HP_Goodman.py`, `BB_HP_Bosch.py`)**: Integration of Goodman and Bosch black-box models; hourly electric consumption for space heating/cooling; COP curves versus ambient and setpoint temperatures.
- **Electric Vehicle (`EMS_EV.py`, `EV_*.py`)**: Battery state-of-charge dynamics with round-trip efficiency; drive cycle and presence modeling; smart charging/discharging with vehicle-to-home/grid capabilities.

**Energy Management Strategy (EMS)**:
- **Base Dispatch (`EMS.py`)**: Numba-JIT compiled real-time hourly economic dispatch for each of 8,760 hours annually. Determines power flow from PV, wind, battery, diesel, and grid to satisfy electrical demand while respecting state-of-charge limits, inverter capacity, grid import/export limits, and minimum diesel loading. Uses marginal-cost-based decision logic with case-by-case prioritization of sources (grid → diesel → battery vs. diesel → grid → battery depending on relative hourly costs).

- **EV Integration (`EMS_EV.py`)**: Extends base dispatch with vehicle availability tracking, dynamic departure planning, and advanced V2X arbitrage. Implements lookahead-based optimization checking next hours for profitable selling opportunities; compares current grid buy price against future sell price thresholds adjusted for battery wear cost and round-trip efficiency losses; incorporates safety buffer dynamics based on historical departure patterns and load variability; and enables vehicle-to-home discharge when economically advantageous and system constraints permit.

- **Heat Pump Allocation (`EMS_HP.py`)**: Post-dispatch energy source attribution using two methods—proportional allocation of grid purchases to heating/cooling/other loads, or marginal-cost-based assignment of all energy sources to heating/cooling based on economic dispatch order. Separates grid costs for heating versus cooling for detailed multi-fuel cost accounting.

**Tariff and Billing Analysis (`Electricity_Bill_Calculator.py`, Rate Calculators)**:
- Implements monthly billing cycles with energy charges, service charges, sales tax, grid taxes, and annual fixed expenses.
- Supports net metering with credit reconciliation (NEM flag); optional grid connection fees; and annual or one-time incentives (renewable energy credits, rebates).
- Flexible rate structure functions for flat, seasonal, monthly, tiered, and TOU pricing; `calcTouRate.py` and `calcULTouRate.py` implement complex real-world tariffs (e.g., California NEM 3.0, ultra-low TOU schemes).
- 25-year financial projection with annual escalation rates, discount factors, and present-value calculations for accurate lifecycle cost analysis.
- Scenario comparison functionality (`compare_scenarios`) for quantifying economic impact of distributed generation and storage versus grid-only operation.

**Financial Analysis (`Advanced_multi_cashflow.py`, `Fitness.py`)**:
- Comprehensive cash-flow modeling: investment costs (with component-specific RECs/incentives), annual replacement costs (pro-rated over remaining lifetime), O&M costs (escalated annually), fuel costs (for DG), and grid/thermal energy costs (with tariff escalation and discounting).
- Net Present Cost (NPC) calculation using capital recovery factor (CRF) with nominal and real discount rates.
- Levelized Cost of Energy (LCOE): annual net costs divided by discounted total energy supplied (including EV charging and thermal energy).
- Levelized Emissions (LEM): total lifecycle emissions (diesel generation + grid-sourced) normalized by supplied energy.

**Load Profile Generation (`generic_load.py`)**:
- Flexible electrical load profile generator supporting multiple input modes for users without detailed load metering.
- Modes include: (i) total annual energy with automatic monthly scaling using predefined seasonal load shapes; (ii) monthly energy totals with month-level scaling; (iii) daily energy profiles with hourly proportional distribution; and (iv) direct hourly CSV input.
- Built-in generic load profiles (July-peaking and January-peaking archetypes) scale to user-specified totals, enabling rapid feasibility studies for residential, commercial, and mixed-use buildings.
- Integration with `daysInMonth.py` for calendar-aware month length handling and `service_charge.py` for tariff administration.

**Metaheuristic Optimizers**:
- **Advanced Differential Evolution (`AdvancedDifferentialEvolution.py`)**: Multiple DE strategies (DE/rand/1, DE/best/1, DE/current-to-best/1, DE/rand/2, DE/best/2) with adaptive F and CR parameters tuned by success rate and iteration progress; heterogeneous initialization (random, center-biased, boundary-biased, Sobol-like) for diverse search space coverage; optional local refinement for promising solutions.
- **Improved Artificial Bee Colony (`ArtificialBeeColony.py`)**: Enhanced employed/onlooker/scout phases with multi-dimensional perturbations, probabilistic selection via exponential fitness scaling, and guided reinitialization of stagnant solutions.
- **Grey Wolf Optimizer (`GreyWolfOptimizer.py`)** and **Particle Swarm (`swarm.py`)**: Alternative population-based strategies for algorithm comparison and benchmarking.

**Fitness Evaluation (`Fitness.py`)**: Central function mapping candidate design vectors (PV capacity, WT count, battery size, DG capacity, inverter rating) to multi-objective performance via hourly dispatch simulation. Computes NPC, LCOE, emissions, reliability (LPSP), and constraint violations (rooftop area, net-metering caps, renewable fraction, budget). Uses smooth penalty functions for constraint handling to improve metaheuristic convergence.

**Utility Functions**:
- `daysInMonth.py`: Calendar utility for leap-year aware month lengths.
- `service_charge.py`: Monthly utility service charge handling.
- `calcMonthlyRate.py`, `calcSeasonalRate.py`, `calcTieredRate.py`, `calcFlatRate.py`, `calcMonthlyTieredRate.py`, `calcSeasonalTieredRate.py`: Modular tariff calculation functions enabling user-defined or pre-defined rate structures.
- `sam_monofacial_poa.py`: Integration with SAM NSRDB weather API for plane-of-array irradiance and temperature.
- `generic_load.py`: Flexible electrical load profile generator supporting multiple input modes—total annual energy with seasonal/monthly/daily scaling factors, or normalized generic profiles (July-peaking vs. January-peaking) scaled to user-specified annual/monthly/daily energy totals. Enables rapid feasibility studies without detailed load metering.
- `EV_travel.py`, `EV_Presence.py`, `EV_demand_dest.py`: Vehicle travel profile generation, home presence tracking, and demand distribution.
- `Ev_Battery_Throughput.py`: EV battery cycling degradation and throughput tracking over project lifetime.
- `top_down.py` and `dataextender.py`: Data interpolation and extension utilities for load and weather time-series.

### Workflow

1. **Input**: User configures `Input_Data.py` with system parameters, load/weather data (including flexible load profile options via `generic_load.py`), tariff structure, and optimization settings.
2. **Optimization Loop**: Metaheuristic optimizer (e.g., ADE) iteratively proposes candidate system designs.
3. **Fitness Evaluation**: For each design candidate, `Fitness.py` calls `EMS.py` (or `EMS_EV.py`/`EMS_HP.py`) to perform 8,760-hour dispatch simulation, accumulates annual costs and emissions, and computes constraint violations.
4. **Financial Projection**: `calculate_electricity_bills()` and `calculate_nyear_projection()` extend year-1 costs to 25-year NPV with escalation and discounting.
5. **Multi-Objective Analysis**: Optimizer explores trade-offs between NPC, LCOE, LEM, and constraints; results are saved to CSV and convergence curves plotted.
6. **Reporting**: `Results.py` generates monthly/annual summaries, hourly time-series, performance plots, and comprehensive financial breakdowns.

# AI Usage Disclosure

Portions of the English-language narrative and code documentation in this JOSS paper were assisted by large language model (LLM) tools (Perplexity AI, January 2026) using the provided SAMA source code and author descriptions as input. LLM assistance was limited to prose generation, editing, and technical description; all software architecture, algorithm design, component modeling, and validation studies remain the work of human authors. Human authors reviewed, edited, and validated all AI-assisted outputs for technical accuracy, originality, and JOSS compliance. Authors remain fully responsible for all submitted materials, including licensing, accuracy, and ethical compliance.

# Availability and Community

SAMA is released under the GNU General Public License v3 (GPL-3.0) and is available on GitHub (https://github.com/[FAST-SAMA-repo]). Source code, documentation, example case studies, and unit tests are included. Contributions, issue reports, and feature requests are welcome from the community. Installation via `pip install sama` and detailed tutorials are provided in the repository README and documentation.

# References

