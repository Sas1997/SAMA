---
title: 'SAMA: Open-Source Multi-Objective Optimization Platform for Hybrid Renewable Energy Microgrid Design'
tags:
  - Python
  - renewable energy
  - microgrid optimization
  - solar photovoltaics
  - battery storage
  - electric vehicles
  - heat pumps
  - metaheuristic algorithms
  - multi-objective optimization
authors:
  - name: Seyyed Ali Sadat
    orcid: 0000-0001-9690-4239
    affiliation: 1
  - name: Joshua M. Pearce
    affiliation: 1
    orcid: 0000-0001-9802-3056
affiliations:
  - index: 1
    name: Department of Electrical & Computer Engineering, Western University, London, ON N6A 3K7, Canada
    ror: 02grkyz14
date: 11 May 2026
bibliography: paper.bib
---

# Summary

SAMA (Solar Alone Multi-Objective Advisor) is a platform for the optimal planning and design of hybrid renewable energy systems for buildings and small communities. From hourly loads, weather data, component costs and the utility's electricity and natural gas tariffs, it sizes the photovoltaic (PV) array, wind turbines, battery bank, generator and inverter. Each design is evaluated by simulating a year of hourly operation with a rule-based energy management system (EMS), computing the bills under the selected tariff and projecting lifecycle cash flows. Electrical and thermal loads are integrated: hourly heating and cooling loads are converted by air-source heat pump models into electricity demand that the system must supply, and an electric vehicle (EV) adds a dynamic load and storage that depend on when it is connected at home and how far it is driven, with vehicle-to-home (V2H) and vehicle-to-grid (V2G) dispatch and price arbitrage. In single-objective mode, one of four metaheuristics minimizes net present cost (NPC), optionally with weighted emissions; in multi-objective mode, NSGA-II [@deb2002nsga2] from pymoo [@blank2020pymoo] approximates the Pareto front between objectives such as NPC, levelized cost of energy, emissions, reliability and renewable fraction. A microgrid here follows the U.S. Department of Energy definition, which concerns electrical boundaries and control rather than size [@ton2012doe]; the [documentation](https://github.com/Sas1997/SAMA/blob/main/Docs/SAMA_Documentation.md) describes residential and community examples.

The core methods of an earlier version, with PV, wind, battery, generator and grid components, Particle Swarm Optimization (PSO) and seven electricity tariffs, were described in @sadat2023sama and compared with the commercial tool HOMER Pro [@lambert2006homer; @homer_energy] for residential systems; that comparison covers the 2023 code. Since then, SAMA has added net metering and internal rate of return (IRR), return on investment (ROI) and payback calculations (2023 to 2024); ultra-low time-of-use (TOU) rates, natural gas tariffs, the EV model with V2H/V2G dispatch and three further optimizers (2025); Bosch and Goodman heat pump models, the pip package, a configuration wizard and a Windows interface (early 2026); and the multi-objective mode, configurable weights, commercial demand charges, bi-facial PV modules, a test suite and continuous integration (late 2026).

# Statement of Need

SAMA was built to go beyond HOMER Pro [@lambert2006homer; @homer_energy], the established commercial tool for sizing hybrid energy systems, while keeping its simulate-and-search approach: each candidate design is operated hour by hour under dispatch rules, and its lifecycle cost is computed from the bills the owner would pay. HOMER Pro, developed at the U.S. National Renewable Energy Laboratory (NREL) and now sold commercially, is closed source and requires a paid license. SAMA is free and open under AGPL-3.0, so every dispatch rule, tariff and cost model can be read, changed and cited, and it adds capabilities that HOMER Pro does not provide (Table 1):

- An EV as a dynamic load and storage, connected according to its arrival and departure times, with V2H and V2G dispatch and price arbitrage built into the sizing; HOMER Pro has no EV component.
- Heating and cooling loads coupled to the electrical load through air-source heat pump models with hourly coefficients of performance from manufacturer data; HOMER Pro does not model heat pumps explicitly.
- Tiered electricity rates and tiered and seasonal natural gas tariffs; HOMER Pro offers constant, hourly and time-of-day electricity rates and a single boiler fuel price.
- An approximate Pareto front between cost, emissions, reliability and renewable fraction; HOMER Pro ranks designs by NPC.
- A pip-installable Python package that can be scripted and extended.

These capabilities matter for the questions SAMA is used to study, such as grid defection, tariff design, net metering and subsidies, whose answers depend on billing details, and for EVs and heat pumps, whose dispatch in homes is still a research question. The intended users are researchers, students and practitioners who need hourly, bill-level sizing of residential, commercial and community systems without a commercial license.

# State of the Field

Most open-source energy system optimization tools are linear or mixed-integer linear programs (MILP). REopt [@cutler2017reopt], developed at NREL and available as the open-source Julia package REopt.jl, sizes and dispatches PV, wind, storage, generators and heating and cooling technologies, including heat pumps, for a site, minimizing lifecycle cost under tariffs with energy and demand charges; it is the closest open-source counterpart to SAMA. SAMA differs from REopt.jl [@reopt_jl] in four ways: REopt.jl minimizes a single objective, lifecycle cost, to which monetized emission costs can be added, whereas SAMA can also approximate the trade-off between cost, emissions, reliability and renewable fraction; REopt.jl optimizes dispatch together with sizing over the year, whereas SAMA operates each design with explicit hourly rules; REopt.jl has no EV model, so V2H, V2G and EV arbitrage cannot be studied with it; and it prices heating fuel as a fixed, monthly or hourly cost rather than through tiered or seasonal gas tariffs. REopt.jl in turn models combined heat and power, ground-source heat pumps and thermal storage, which SAMA does not. MicroGridsPy [@balderrama2019microgridspy] sizes village-scale microgrids with linear programming; its earlier Pyomo version also offered multi-objective optimization of NPC and CO$_2$ emissions, which the current package does not yet expose. oemof.solph [@krien2020oemof], PyPSA [@brown2018pypsa] and Calliope [@pfenninger2018calliope] are general frameworks for building linear and MILP energy system models, and OSeMOSYS [@howells2011osemosys] targets long-term national and regional planning. pvlib python [@anderson2023pvlib] models PV performance, and NREL's SAM [@nrel_sam] and PVWatts [@pvwatts] and the proprietary PVsyst [@pvsyst] and RETScreen [@retscreen] are mainly performance and financial simulators of a given design; SAM can also size the battery of a user-designed PV-battery system through REopt [@sam_reopt].

In the programming-based tools, dispatch is a decision variable optimized together with sizing, either over the whole modeled period with perfect foresight or, in frameworks such as PyPSA, over rolling horizons with limited foresight [@pypsa_rolling]. Many of the relations in the simulation and its cost are nonlinear, discontinuous or conditional: the lead-acid battery kinetics that limit charge and discharge power, tiered bills whose price depends on the cumulative monthly consumption, annual net-metering reconciliation, and generator replacements that depend on its operating hours. In these tools they must be written as linear or mixed-integer constraints, some exactly and some only by approximation. SAMA instead treats the yearly simulation as a black box: dispatch follows explicit rules hour by hour, as in HOMER Pro, each bill is computed as the tariff defines it, and derivative-free metaheuristics search the design space. In exchange, a new dispatch rule or tariff is added as ordinary Python code rather than as a new mathematical formulation; the cost is that metaheuristic search needs thousands of simulations per run and cannot prove that its best design is the global optimum, so runs can be repeated with different random seeds to check that the result is stable. The approaches are complementary: programming-based tools give the optimal dispatch for their formulation and information horizon, while SAMA gives the cost of a design operated by explicit rules.

Table 1 compares the capabilities of SAMA, HOMER Pro and the open-source sizing tools. HOMER Pro also offers built-in sensitivity analysis and combined heat and power, which SAMA does not.

| Capability | SAMA | HOMER Pro | REopt | MicroGridsPy | oemof.solph, PyPSA, Calliope |
|:---|:---:|:---:|:---:|:---:|:---:|
| Open source and free | Yes | No | Yes | Yes | Yes |
| Python package | Yes | No | No (Julia) | Yes | Yes |
| Component sizing optimization | Yes | Yes | Yes | Yes | Yes |
| Built-in rule-based hourly dispatch | Yes | Yes | No | No | No |
| EV with V2H and V2G dispatch | Yes | No | No | No | User-built |
| Heat pump with hourly COP | Yes | No | Yes | No | User-built |
| Multi-objective Pareto front | Yes | No | No | Partial | User-built |
| Tiered electricity rates | Yes | No | Yes | No | User-built |
| Time-of-use electricity rates | Yes | Yes | Yes | Yes | User-built |
| Natural gas tariffs (tiered, seasonal) | Yes | No | Partial | No | User-built |
| Net metering with annual reconciliation | Yes | Yes | Partial | No | User-built |
| Demand charges | Yes | Yes | Yes | No | User-built |
| IRR, ROI and payback | Yes | Yes | Partial | No | No |

Table: Capabilities as stated in SAMA's source code and in each tool's public documentation (checked November 2026). HOMER Pro has no EV component, prices grid power as constant, hourly or time-of-day and monthly rates without consumption tiers, prices boiler fuel at a single value, does not model heat pumps explicitly, and seeks the least-cost system, ranking results by NPC; HOMER Grid models EV charging stations, without V2G. REopt.jl has no EV input and optimizes dispatch; it caps net-metered export at annual grid purchases, takes fuel cost as a scalar, monthly or hourly value without tiers, and reports IRR and payback without ROI ("Partial"). MicroGridsPy (current package) models renewables, batteries, generators and the grid, buys and sells grid power at hourly import and export prices, and reports NPC and LCOE; its multi-objective optimization was part of the earlier Pyomo version and is not exposed in the current package ("Partial"). "User-built" means the framework can represent the feature but the user must formulate it; their economic outputs are costs and revenues, without IRR, ROI or payback.

SAMA's most distinctive feature is its EV energy management. Linear frameworks usually represent an EV as a storage unit or flexible load optimized over the horizon. SAMA embeds a rule-based V2H/V2G controller in the sizing loop: it finds the next departure, charges from renewable surplus first, charges from the grid in low-price hours when the sell prices before departure exceed the purchase price by a set margin after round-trip losses (arbitrage), sells only when the price covers the EV battery wear cost, and keeps a safety buffer so the required departure state of charge is met. Its look-ahead uses the tariff and the simulated PV, wind and load until the next departure and ends there.

SAMA was started in 2022, in MATLAB and Python, to bring HOMER Pro's simulate-and-search approach into open code for residential studies [@sadat2023sama]. Building it on one of the MILP frameworks would have replaced its rule-based dispatch with optimized dispatch, which changes the method rather than extending it.

# Software Design

SAMA has six subsystems: input configuration, component models, the hourly EMS, tariff and financial calculators, optimizers, and reporting. All inputs are held in one configuration object that can be filled from a YAML file, so a study needs no source edits.

**Search method.** Each evaluation simulates 8,760 hours and returns NPC, levelized cost of energy (LCOE), levelized emissions (LEM), loss of power supply probability (LPSP), renewable fraction and capital cost. With component replacements, the generator's minimum load, tiered and TOU bills and annual net-metering reconciliation, the objective is non-smooth and has no analytical gradient, so SAMA uses population-based metaheuristics. Integer variables such as the number of battery packs are rounded inside the evaluation, and runs can be repeated with different seeds (`Run_Time`).

**Performance.** In SAMA's EMS, each hour's dispatch branches on the battery and EV states left by the hours before, so the yearly loop is sequential. An early contributor version vectorized the dispatch with NumPy; the current EMS, battery, EV and heat pump kernels keep the sequential loop and compile it with Numba. With the default inputs, one fitness evaluation (dispatch, bills and lifecycle cost) takes a median of about 2.5 ms compiled and 170 ms with Numba disabled (Python 3.13, Numba 0.68, one core of a 2.8 GHz Intel Xeon; script `benchmarks/fitness_timing.py`), so a default run of about 10,000 evaluations takes under a minute instead of about half an hour. The costs are a compilation delay on first use and the restricted Python allowed in compiled code.

**Optimizers.** PSO [@kennedy1995particle; @shi1998modified] was written for the 2022 to 2023 MATLAB and Python versions and the Grey Wolf Optimizer (GWO) [@mirjalili2014grey] for the MATLAB version; Advanced Differential Evolution (ADE) [@storn1997differential; @das2011differential], Artificial Bee Colony (ABC) [@karaboga2007powerful] and the Python GWO were added in 2025. They are not new algorithms. They share one interface for settings, seeds, bounds and convergence output and call the SAMA fitness directly, so algorithms can be compared on the same design problem. The multi-objective mode uses pymoo rather than a new implementation.

**Objectives and constraints.** The single-objective fitness is `NPC_weight * NPC + EM_weight * EM * LEM` plus smooth penalties for violated constraints. `EM` is 0 by default, giving NPC alone. The default weights, $10^{-5}$ for NPC in dollars and $10^{2}$ for LEM in kg/kWh, only bring the terms to similar magnitudes and are set in the configuration file. A fixed weighted sum returns one design and cannot reach non-convex parts of a trade-off, which is why the multi-objective mode was added. There, constraints are passed to NSGA-II as inequality constraints, so feasible designs are always preferred. Because many designs differ negligibly in some objective, such as the sixth decimal of the renewable fraction, the front is filtered by epsilon-dominance with a minimum resolution per objective. The front is written to a CSV file and a figure, and one design is selected for the detailed reports: a compromise (achievement scalarizing function on the normalized objectives), the best design for one objective, or a weighted choice.

**Models.** PV output uses a temperature-corrected efficiency and plane-of-array irradiance, with weather from the National Solar Radiation Database [@nsrdb] or user files; for bi-facial modules, the rear irradiance from the infinite sheds view-factor model of pvlib [@anderson2023pvlib] is added with the module bifaciality, and the array can be split into sub-arrays of different orientation. Wind turbines use a power-law shear profile and a power curve. Lithium-ion batteries use constant efficiency with throughput-based degradation, and lead-acid batteries the Kinetic Battery Model [@kibam]. Heat pumps use Bosch and Goodman manufacturer tables for hourly coefficients of performance. In each hour, the base EMS meets the net load by ranking battery, generator and grid by marginal cost in six dispatch cases; the EV EMS extends it. Eight electricity rate structures (flat, seasonal, monthly, tiered, seasonal tiered, monthly tiered, TOU and ultra-low TOU) and eight natural gas structures are separate calculators [@sadat2025pricing]; commercial demand charges (flat, seasonal, monthly or TOU, with an optional ratchet) are billed on the monthly peaks of the grid purchases, together with an optional minimum charge rule. The financial model includes real and nominal discounting, price escalation, replacements, incentives and net metering, with PV costs following NREL benchmarks [@nrel_costs].

**Interfaces and testing.** `pip install samapy` provides `samapy-config`, an interactive wizard, and `samapy-run`, which runs a YAML configuration in either mode. SAMA can also be run from the repository [@sama_github], through its Python API, or with an alpha Windows interface [@SAMA_alpha_win]. A pytest suite, run on GitHub Actions for Python 3.10 and 3.12, checks inputs, the energy balance, the optimizers against known optima, the multi-objective mode against an analytical Pareto front, and the command-line workflow.

# Research Impact Statement

Seven peer-reviewed studies have used SAMA. The 2023 paper describes and validates the 2023 code [@sadat2023sama]. Five of the others used the PV, wind, battery, generator and grid sizing core with the tariff and financial calculators to study grid defection thresholds of U.S. households [@sadat2024defection], peer-to-peer virtual net metering with co-authors outside the core team [@groza2024nem], electricity pricing structures in Canada [@sadat2025pricing], PV investments as inflation hedges [@sadat2025inflation], and redirecting Alberta's oil and gas subsidies to solar power [@sadat2026alberta]. The pricing study led directly to the net-metering model and the return on investment, internal rate of return and payback calculations. These six were published before the Bosch and Goodman heat pump models, the pip package and the multi-objective mode were released, and none used the EV or heat pump modules. The seventh, a study of agrivoltaic shield walls in Florida with co-authors outside the core team [@hayibo2026agrivoltaic], used bi-facial modules and commercial demand charges, which were developed for it and then added to the package. Development has also drawn on contributors who wrote tariff calculators and an early NumPy version of the EMS, and on users: an external bug report led to a corrected energy balance in the dispatch engine. The package is on PyPI and the source on GitHub [@sama_github].

# AI Usage Disclosure

Generative AI tools (Claude models, Anthropic, 2025-2026) were used in a supporting role. The optimization methods, system and component models, energy management strategies, and algorithms in SAMA were designed and implemented by the authors without generative AI. AI tools assisted with debugging and troubleshooting existing code, generating figure plotting code, developing parts of the Windows graphical interface, drafting and editing the user documentation, converting the documentation from Word to Markdown, and editing portions of this paper's prose for clarity. The authors reviewed, tested, and validated all AI-assisted outputs for technical accuracy and originality, and take full responsibility for all submitted materials.

# Acknowledgements

This work was supported by the Natural Sciences and Engineering Research Council of Canada (NSERC) and the Thompson Endowment.

# References
