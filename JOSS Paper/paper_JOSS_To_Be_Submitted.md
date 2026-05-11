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

Solar Alone Multi-Objective Advisor (SAMA) is an open-source platform for the planning, design, and operation optimization of hybrid renewable energy microgrids. Given location-specific electrical and thermal load profiles, meteorological data, and utility billing structures, SAMA finds the optimal combination of user-selected system components, including solar photovoltaic (PV) capacity, wind turbines, battery storage, backup generator, and inverter rating, that minimizes the lifecycle net present cost (NPC) of the system. Users choose which components to include in the optimization (e.g., PV-only, PV with battery storage, or full PV-battery-generator hybrid), and may optionally include a second objective, e.g., levelized emissions (LEM), to jointly optimize cost and environmental impact.

The platform simulates all hours of a representative year through an hourly economic dispatch engine, supports electric vehicles (EVs) with vehicle-to-home (V2H) and vehicle-to-grid (V2G) capabilities (user's choice), integrates air-source heat pump models (user's choice), and handles eight distinct electricity tariff structures including time-of-use (TOU) and tiered pricing. Four metaheuristic algorithms are available for the optimization: Particle Swarm Optimization (PSO), Advanced Differential Evolution (ADE), Artificial Bee Colony (ABC), and Grey Wolf Optimizer (GWO). SAMA is distributed as a pip-installable Python package, is available for direct download and execution from the GitHub repository [@sama_github], and has been validated against HOMER Pro across multiple U.S. climate zones [@sadat2023sama].

# Statement of Need

The global deployment of solar photovoltaic (PV) systems and hybrid renewable energy microgrids is accelerating rapidly, yet the software tools needed to optimally size and economically evaluate these systems remain inaccessible to much of the world [@sadat2023sama]. Designing a hybrid system requires simultaneous optimization over competing objectives (minimizing lifecycle cost, meeting reliability constraints, maximizing renewable penetration) while accounting for location-specific meteorology, load profiles, and utility billing structures. Without appropriate software, practitioners resort to oversized pr undersized systems, underperforming designs, or abandon renewable deployment entirely.

The dominant tool for microgrid planning and design, HOMER Pro, was originally developed as open-source software by the U.S. National Renewable Energy Laboratory (NREL) but has transitioned to a closed-source, for-profit model [@homer_energy]. Licensing costs range from \$187.50 to \$568.50 per month for standard users, placing it out of reach for low-resource laboratories, researchers in developing regions, small municipalities, NGOs, and independent practitioners. Beyond cost, its closed-source nature prevents researchers and developers from inspecting dispatch logic, adapting billing structures, or extending the software to novel or emerging technologies. This is a fundamental barrier: as hybrid systems increasingly incorporate electric vehicles (EVs), heat pumps, thermal batteries, peer-to-peer net metering, and novel tariff structures, the inability to modify or audit HOMER Pro's internal models has become a critical limitation for rigorous research and reproducible science.

Open-source alternatives exist but remain narrowly scoped. Most PV simulation tools, including SAM [@nrel_sam], PVWatts [@pvwatts], PVsyst [@pvsyst], and RETScreen [@retscreen], provide performance simulation or financial analysis for individual technologies but do not perform multi-component hybrid system optimization. None implement the complex utility tariff structures needed for real-world economic analysis of grid-interactive systems, and none support EVs or heat pumps. SAMA was developed specifically to fill this gap: providing a fully open, GPL-3.0-licensed, validated alternative to HOMER Pro that extends its capabilities to emerging load types, system components and tariff regimes while remaining freely accessible to the global research community [@sadat2023sama; @sadat2025pricing].

# State of the Field

A comprehensive survey of available PV and hybrid energy system software tools was conducted by Sadat et al. [@sadat2023sama], covering different tools across simulation, design, and optimization categories. The findings are summarized here to contextualisecontextualize SAMA's contribution.

Tools such as the NREL System Advisor Model (SAM) [@nrel_sam] enable exploration of the financial and physical parameters of solar and storage systems, but focus on single-technology performance simulation rather than multi-component hybrid optimization. PVWatts [@pvwatts], developed by NREL, estimates energy production and cost savings for grid-connected PV systems and is widely used for preliminary evaluations, but does not support hybrid system sizing. RETScreen (Natural Resources Canada) [@retscreen] assists in energy generation and conservation decisions across diverse project types, including hybrid wind-PV-generator configurations, but does not perform optimal sizing of hybrid energy systems. PVsyst [@pvsyst] provides detailed simulation of grid-connected, stand-alone, pumping, and DC-grid PV systems with a comprehensive meteorological and component database, but again without optimization capability for hybrid multi-source configurations. TRNSYS [@trnsys], a modular transient systems simulation environment developed at the University of Wisconsin, is highly flexible for dynamic energy system analysis but requires significant user expertise and does not include built-in HES optimization.

As demonstrated in the comparative table of Sadat et al. [@sadat2023sama], HOMER Pro [@homer_energy] and HOMER Grid [@homer_grid] are the only tools in the field that provide optimal sizing of hybrid energy systems. HOMER Pro , originally developed as open-source software by NREL, offers optimization of PV, wind, battery, diesel generator, grid, hydrogen, hydro, and biomass combinations with cost sensitivity and emissions analysis. HOMER Grid is designed to optimize the value of behind-the-meter distributed energy systems for commercial grid-connected applications. Both remain the dominant commercial tools in the field; however, they are closed-source, costly, and cannot be adapted or extended by users, a fundamental limitation for research reproducibility and for modeling emerging technologies.

SAMA's key differentiators within this landscape are: (1) full source transparency under AGPL-3.0 licensing, enabling users to inspect, adapt, and extend all dispatch logic, tariff structures, and component models, which is a fundamental requirement for reproducible research that no commercial alternative provides [@sama_github]; (2) comprehensive dynamic tariff and lifecycle economic modeling covering various electricity and natural gas rate structures, including TOU, tiered, seasonal, and ultra-low TOU regimes, with annual price escalation, adjustments and taxations, and renewable incentive modeling [@sadat2025pricing]; (3) a novel EV energy management system with departure-aware scheduling, three-tier V2G arbitrage, renewable surplus prioritization, and adaptive safety buffering; (4) integrated heat pump performance models using manufacturer lookup data for hourly COP estimation; (5) user-configurable system topology, allowing optimization of any combination of PV, wind, battery, diesel generator, EV, and heat pump components; and (6) four interchangeable metaheuristic optimizers (PSO, ADE, ABC, GWO) enabling algorithmic comparison within a consistent evaluation framework.

# Software Design

SAMA is structured as a modular Python package with six subsystems: core input configuration, component physics models, an hourly energy management system (EMS), tariff and billing calculators, metaheuristic optimizers, and result reporting. Key design decisions included prioritizing numerical performance for the inner dispatch loop (addressed through Numba JIT compilation), separating tariff logic into independent calculator modules to accommodate diverse utility structures, and using a singleton input configuration object to allow external YAML-based configuration without modifying source code. Full technical documentation is provided in the repository [@sama_github].

**Component models.** The solar PV model applies a temperature-dependent efficiency formulation with nominal operating cell temperature (NOCT) correction using plane-of-array irradiance from the National Solar Radiation Database (NSRDB) [@nsrdb]. The wind turbine model applies a power-law wind shear profile and a piecewise power curve. Battery storage supports lithium-ion (constant-efficiency model with lifetime throughput degradation) and lead-acid chemistryies via the Kinetic Battery Model (KiBaM) [@kibam]. The diesel generator uses a linear fuel consumption curve. Heat pump performance is represented by manufacturer lookup tables for Bosch and Goodman systems, returning hourly COP values as functions of ambient temperature and building thermal load. The EV model tracks state-of-charge dynamics with round-trip efficiency.

**Energy management system.** The base EMS (`EMS.py`) is Numba JIT-compiled for performance. In each of annual hours, the dispatcher resolves net load using marginal-cost-based prioritization across the battery, diesel generator, and utility grid. Six dispatch scenarios are defined from the relative values of the grid buy price, diesel cost, and battery wear cost, enabling cost-optimal decisions at each time step. Unmet energy accumulates toward the loss of power supply probability (LPSP) constraint.

The EV extension (`EMS_EV.py`) manages V2H and V2G dispatch through a departure-aware scheduler that dynamically identifies the next departure time, prioritizes renewable surplus for required charging, and opportunistically pre-charges from the grid during low-price windows. For V2G, a three-tier arbitrage engine selects between aggressive, moderate, and conservative discharge strategies based on price percentile thresholds, profit margin requirements, time-to-departure flexibility, and inverter capacity. A rolling safety buffer derived from historical departure data ensures the EV always meets its required departure SOC, and all discharge decisions undergo final net-profit validation accounting for battery wear cost, round-trip efficiency losses, and solar displacement penalties. The heat pump post-processor (`EMS_HP.py`) attributes hourly electricity consumption to heating versus cooling loads using proportional and marginal-cost-based methods.

**Tariff and financial modeling.** SAMA implements eight electricity rate structures and eight natural gas rate structures, covering flat, seasonal, monthly, tiered, TOU, and ultra-low TOU tariffs used across North American utilities [@sadat2025pricing]. Net metering is modeled with annual credit reconciliation. The lifecycle financial projection computes NPC using the capital recovery factor (CRF) with real and nominal discount rates, applies annual price escalation for grid electricity and natural gas, schedules component replacements, and applies renewable energy incentives (such as Investment Tax Credits (ITCs)). PV system cost benchmarks follow NREL methodology [@nrel_costs].

**Optimization framework.** The five design variables (PV capacity, wind turbine count, battery count, diesel generator size, and inverter rating) are explored by any of the four included metaheuristic algorithms. ADE uses different mutation strategies with adaptive scaling and crossover parameters [@storn1997differential; @das2011differential]. ABC uses multi-dimensional perturbations with guided scout reinitialization [@karaboga2007powerful]. GWO implements social hierarchy convergence with an optional parallel multiprocessing variant [@mirjalili2014grey]. PSO uses inertia-weight damping with cognitive and social learning coefficients [@kennedy1995particle; @shi1998modified]. All four algorithms share the same fitness function and smooth constraint penalty framework, enabling direct performance comparisons.

**Interfaces and access.** SAMA can be accessed in three ways. First, it is installable via `pip install samapy` and provides two command-line entry points: `samapy-config` (an interactive configuration wizard) and `samapy-run` (an optimization runner accepting YAML configuration files). Second, users can download or clone the complete source code directly from the GitHub repository at [https://github.com/Sas1997/SAMA](https://github.com/Sas1997/SAMA) [@sama_github] and execute SAMA using any Python-compatible IDE such as PyCharm, or from the command line in any Python 3.9+ environment. Third, a graphical user interface (GUI) based Windows executable (.exe) currently on alpha development stage is available for users who prefer not to install a Python environment; this is distributed on GitHub [@SAMA_alpha_win]. A Python API also enables programmatic access for integration with external workflows. Comprehensive user documentation, including installation instructions, input file specifications, and worked examples, is provided in the repository documentation [@sama_github].

# Research Impact Statement

SAMA has supported six peer-reviewed publications. The primary paper describes the core platform and validation against HOMER Pro across Sacramento, California and New Bern, North Carolina, demonstrating close agreement in optimal system configurations, LCOE, NPC, and battery state-of-charge profiles [@sadat2023sama]. Subsequent studies applied SAMA to analyze economic grid defection thresholds for U.S. households [@sadat2024defection], evaluate the impact of electricity pricing structures on PV and PV-battery systems across Canadian utilities [@sadat2025pricing], quantify PV systems as inflation hedges [@sadat2025inflation], model peer-to-peer virtual net metering schemes [@groza2024nem], and analyze subsidy redirection scenarios in Alberta [@sadat2026alberta]. The package is indexed on PyPI and the source repository is publicly accessible on GitHub [@sama_github].

# AI Usage Disclosure

Generative AI tools (Claude Sonnet 4.5, Anthropic, 2025-2026) were used to assist with editing and restructuring the user documentation, converting documentation from Word to Markdown format, and editing portions of this paper's prose for clarity. Human authors reviewed, edited, and validated all AI-assisted outputs for technical accuracy, originality, and compliance with JOSS requirements. The authors bear full responsibility for the accuracy, originality, and ethical compliance of all submitted materials.

# Acknowledgements
This work was supported by the Natural Sciences and Engineering Research Council of Canada (NSERC) and the Thompson Endowment.

# References
