<p align="center">
  <img src="https://raw.githubusercontent.com/Sas1997/SAMA/main/Assets/SAMA_Logo-with_Typography.png" alt="SAMA Logo" width="420"/>
</p>

<h2 align="center">Solar Alone Multi-Objective Advisor</h2>
<h4 align="center">Open-Source Hybrid Renewable Energy Microgrid Optimization</h4>

<p align="center">
  <a href="https://doi.org/10.1016/j.enconman.2023.117686"><img src="https://img.shields.io/badge/DOI-10.1016%2Fj.enconman.2023.117686-blue?style=flat-square" alt="DOI"/></a>
  <a href="https://github.com/Sas1997/SAMA/blob/main/LICENSE"><img src="https://img.shields.io/badge/License-AGPL%20v3-green?style=flat-square" alt="License: AGPL v3"/></a>
  <a href="https://pypi.org/project/samapy/"><img src="https://img.shields.io/pypi/v/samapy?style=flat-square&label=PyPI&color=orange" alt="PyPI"/></a>
  <img src="https://img.shields.io/badge/Python-3.9%2B-blue?style=flat-square&logo=python&logoColor=white" alt="Python 3.9+"/>
  <a href="https://github.com/Sas1997/SAMA"><img src="https://img.shields.io/github/stars/Sas1997/SAMA?style=flat-square&color=yellow" alt="GitHub Stars"/></a>
</p>

---

**SAMA** is a free, open-source Python platform for the design, sizing, and operation optimization of hybrid renewable energy microgrids. It performs location-specific techno-economic assessment using site-dependent load profiles, meteorological data, component degradation models, and utility billing structures, with no license fees and no black-box code.

Developed by the **[FAST Research Group](https://github.com/Sas1997/SAMA)** at **Western University**, London, Ontario, Canada.

---

## Table of Contents

- [Why SAMA?](#why-sama)
- [What SAMA Optimizes](#what-sama-optimizes)
- [Key Capabilities](#key-capabilities)
- [Getting Started](#getting-started)
  - [Method 1: Raw Python Code](#method-1-raw-python-code)
  - [Method 2: Python Package (pip)](#method-2-python-package-pip)
  - [Method 3: Windows .exe (Alpha)](#method-3-windows-exe-alpha)
- [Quick Start](#quick-start)
- [Optimization Modes](#optimization-modes)
- [Optimization Algorithms](#optimization-algorithms)
- [Output Examples](#output-examples)
- [Documentation](#documentation)
- [Running the Tests](#running-the-tests)
- [Publications](#publications)
- [Citation](#citation)
- [License](#license)
- [Contact](#contact)

---

## Why SAMA?

Proprietary tools like HOMER Pro and HOMER Grid are closed-source and license-restricted. SAMA was built to remove that barrier:

- **Free and open-source**: AGPL-3.0
- **Transparent algorithms**: read, modify, and extend every line
- **Validated**: the 2023 version was compared with HOMER Pro for residential systems in Sacramento, California and New Bern, North Carolina ([Sadat et al., 2023](https://doi.org/10.1016/j.enconman.2023.117686)); that comparison has not been repeated for the current version
- **No cost barriers**: designed for researchers, students, and resource-constrained institutions worldwide

---

## What SAMA Optimizes

SAMA finds the optimal combination of five system design variables by minimizing **Net Present Cost (NPC)** over a 25-year project lifetime:

| Variable | Description |
|----------|-------------|
| `Npv` | Number of PV modules (kW of rated capacity) |
| `Nwt` | Number of wind turbines |
| `Nbat` | Number of battery packs |
| `N_DG` | Number / size of diesel generator units |
| `Cn_I` | Inverter capacity (kW) |

Constraints include reliability (LPSP), renewable energy fraction, capital budget, and rooftop area limits. SAMA can also return the trade-off between objectives such as cost, emissions and renewable fraction as a Pareto front (see [Optimization Modes](#optimization-modes)).

---

## Key Capabilities

| Category | Details |
|----------|---------|
| **System Components** | Solar PV (mono- or bi-facial modules, one or more sub-arrays), wind turbines, battery storage (Li-ion or lead-acid), diesel generator, utility grid, air-source heat pump (Bosch or Goodman), electric vehicle with V2X |
| **Optimization Algorithms** | Particle Swarm Optimization (PSO), Advanced Differential Evolution (ADE), Artificial Bee Colony (ABC), Grey Wolf Optimizer (GWO) |
| **Energy Simulation** | 8,760-hour hourly dispatch with marginal-cost-based EMS, EV smart charging and V2X lookahead arbitrage, heat pump thermal-electric coupling. The EV and heat pump are simulated within every design the optimizer evaluates, so the system is sized with them integrated |
| **Electricity Tariffs** | 8 structures: flat, seasonal, monthly, tiered, seasonal tiered, monthly tiered, Time-of-Use (TOU), Ultra-Low TOU (ULO); commercial and industrial demand charges (flat, seasonal, monthly or TOU, with optional ratchet) and a minimum charge rule |
| **Natural Gas Tariffs** | 8 structures including PG&E G-1 therms-based and Enbridge EGD 4-tier m³-based |
| **Financial Analysis** | 25-year NPC, LCOE, levelized emissions, price escalation, RE incentives (30% ITC), net metering reconciliation |
| **Load Inputs** | 10 electrical load input modes (CSV, monthly/annual averages, generic profiles), thermal load from Excel |
| **Weather Inputs** | SAM NSRDB API integration, user CSV files, monthly/annual averages |
| **Outputs** | Convergence plots, 25-year cash flow charts, energy distribution, battery SOC, grid interconnection, EV and heat pump performance charts, hourly CSV data |

---

## Getting Started

SAMA can be used in three ways:

### Method 1: Raw Python Code
For researchers and developers who want full access to source code and algorithms.
```bash
git clone https://github.com/Sas1997/SAMA.git
cd SAMA
pip install -e .
```
This installs SAMA in editable mode, so you can edit any file under `src/samapy/` directly and see your changes immediately. Configure your system with the wizard or by editing `src/samapy/core/Input_Data.py`, then run:
```bash
samapy-run
```
Available algorithms: `pso`, `ade`, `abc`, `gwo`
### Method 2: Python Package (pip)
The recommended method for most users. No source editing required.
```bash
pip install samapy
```
> **Note:** The PyPI distribution name is `samapy`. Once installed, import and CLI commands use `samapy`.
**Step 1: Configure your project:**
```bash
samapy-config
```
An interactive multi section wizard guides you through all parameters and saves `samapy_config_COMPLETE.yaml`.
**Step 2: Run the optimization:**
```bash
samapy-run
```
**Step 3: View results in `samapy_outputs/`**

Additional run options:

```bash
samapy-run --algorithm ade          # override algorithm
samapy-run --output results/run1    # override output directory
samapy-run --dry-run --verbose      # validate config without running
samapy-run --no-gui                 # headless / server mode
samapy-run --seed 1                 # reproducible run (fixed random seed)
samapy-run --mode multi             # multi-objective mode (Pareto front)
```

**Reference example.** [`examples/README.md`](https://github.com/Sas1997/SAMA/blob/main/examples/README.md) maps every way of running SAMA to its inputs, configuration and outputs, and [`examples/reference_config.yaml`](https://github.com/Sas1997/SAMA/blob/main/examples/reference_config.yaml) is the reference case whose full results are in [Section 18 of the documentation](https://github.com/Sas1997/SAMA/blob/main/Docs/SAMA_Documentation.md#18-sample-optimization-output):

```bash
samapy-run -c examples/reference_config.yaml --seed 1 --no-gui --output reference_run
```

### Method 3: Windows .exe (Alpha)

A point-and-click GUI for non-technical users. No Python installation required. Currently in alpha.

> [https://github.com/Sas1997/SAMA](https://github.com/Sas1997/SAMA/releases/latest/download/SAMA-Setup.exe)

---

## Quick Start

Run SAMAPY in five steps using bundled sample data (Boston, MA):

```bash
# 1. Install
pip install samapy

# 2. Create a project folder and navigate into it
mkdir my_samapy_project && cd my_samapy_project

# 3. Launch the configuration wizard (accept defaults for a quick test)
samapy-config

# 4. Run the optimization
samapy-run

# 5. View your results
#    samapy_outputs/figs/    → all charts (PNG/SVG)
#    samapy_outputs/data/    → Outputforplotting.csv (8,760-hour time series)
```

> **Expected runtime:** 2–10 minutes for `MaxIt=200`, `nPop=50` depending on hardware. The first run is slower due to one-time Numba JIT compilation of the EMS engine.

---

## Optimization Modes

SAMA can be run in two modes:

**Single-objective mode** (default, `optimization_mode: single`). One of the four algorithms below minimizes a scalar fitness:

```
fitness = NPC_weight * NPC + EM_weight * EM * LEM + constraint penalties
```

With `EM = 0` (default) this is net present cost alone. With `EM = 1` levelized emissions are added with the weight `EM_weight`. `NPC_weight` (default `1e-5`) and `EM_weight` (default `1e2`) only put the two quantities on comparable numerical scales; both can be set in the configuration file. A weighted sum with fixed weights returns one design, and it can only reach designs on the convex part of the cost-emissions trade-off.

**Multi-objective mode** (`optimization_mode: multi`, or `samapy-run --mode multi`). NSGA-II from the [pymoo](https://pymoo.org) library searches for the Pareto front: the set of designs for which no objective can be improved without worsening another. Choose two or more objectives in `MOO_objectives` from `npc`, `lcoe`, `lem` (emissions), `lpsp` (unmet demand) and `re` (renewable fraction, maximized). Reliability, minimum renewable fraction, budget and the capacity limits are handled as constraints, so the trade-off is explored within those limits.

```yaml
optimization_mode: multi
MOO_objectives: [npc, re]       # cost versus renewable fraction
MOO_selection: compromise       # or min_npc, min_re, ..., or weights such as [2, 1]
MOO_n_pop: 50                   # optional, default nPop
MOO_n_gen: 200                  # optional, default MaxIt
```

The run writes `pareto_front.csv` (every Pareto-optimal design with its NPC, LCOE, LEM, LPSP, renewable fraction and capital cost) and `pareto_front.png`. One design is then selected for the full SAMA results: `compromise` takes the design closest to the ideal point with equal weight on each normalized objective, `min_<objective>` takes the best design for one objective, and a list of weights favours the objectives with larger weights. Grid emission factors (`E_CO2`, `E_SO2`, `E_NOx`) are zero in the example configuration, so set them before using `lem` as an objective.

---

## Optimization Algorithms

In single-objective mode, all four algorithms optimize the same 5-variable design space. Choose based on your preference for convergence speed vs. exploration depth:

| Algorithm | Key Behavior | Best For |
|-----------|-------------|----------|
| **PSO**: Particle Swarm Optimization | Velocity-based swarm convergence | Fast convergence |
| **ADE**: Advanced Differential Evolution | Multi-strategy mutation with adaptive F and CR | Thorough exploration |
| **ABC**: Artificial Bee Colony | Foraging phases with guided reinitialization | Balanced exploration |
| **GWO**: Grey Wolf Optimizer | Pack hierarchy convergence; parallel variant available | Fast convergence, parallel runs |

For rigorous comparison, set `Run_Time > 1` to average across multiple independent runs.

---

## Output Examples

After `samapy-run` completes, results are organized in `samapy_outputs/`:

```
samapy_outputs/
├── Optimization.png                         ← convergence curve
├── pareto_front.csv, pareto_front.png       ← Pareto front (multi-objective mode)
├── figs/
│   ├── Cash_Flow.svg                        ← 25-year lifecycle cost breakdown
│   ├── Energy Distribution.png              ← hourly served load and supply by source
│   ├── Energy Balance.png                   ← hourly dumped energy and shortages
│   ├── Grid Interconnection.png             ← hourly grid import/export
│   ├── electricity_comparison_annual.png    ← system vs. grid-only cost, by year
│   ├── electricity_comparison_monthly.png   ← system vs. grid-only cost, by month
│   ├── Battery State of Charge.png          ← hourly SOC (if a battery is selected)
│   ├── hp_monthly_summary.png               ← heat pump COP and energy (if HP=1)
│   └── EV Energy.png                        ← EV charge/discharge (if EV=1)
└── data/
    └── Outputforplotting.csv                ← full 8,760-hour time series
```

---

## Documentation

Full documentation is available in this repository:

- **[SAMA_Documentation.md](https://github.com/Sas1997/SAMA/blob/main/Docs/SAMA_Documentation.md)**: Complete user guide covering all parameters, algorithms, financial models, input formats, troubleshooting, and mathematical formulations

Key documentation sections:

| Topic | Location |
|-------|----------|
| All configuration parameters | [Section 6: Configuration File](https://github.com/Sas1997/SAMA/blob/main/Docs/SAMA_Documentation.md#6-the-configuration-file-sama_config_completeyaml) |
| Optimization algorithms (detail) | [Section 7](https://github.com/Sas1997/SAMA/blob/main/Docs/SAMA_Documentation.md#7-optimization-algorithms) |
| Energy management system | [Section 8](https://github.com/Sas1997/SAMA/blob/main/Docs/SAMA_Documentation.md#8-energy-management-system-ems) |
| Component models | [Section 9](https://github.com/Sas1997/SAMA/blob/main/Docs/SAMA_Documentation.md#9-component-models) |
| Financial model | [Section 10](https://github.com/Sas1997/SAMA/blob/main/Docs/SAMA_Documentation.md#10-financial-model) |
| Troubleshooting | [Section 13](https://github.com/Sas1997/SAMA/blob/main/Docs/SAMA_Documentation.md#13-troubleshooting) |
| Mathematical formulations | [Section 19](https://github.com/Sas1997/SAMA/blob/main/Docs/SAMA_Documentation.md#19-technical-framework-and-mathematical-formulation) |
| Where to find input data | [Appendix A](https://github.com/Sas1997/SAMA/blob/main/Docs/SAMA_Documentation.md#appendix-a-where-to-find-input-data-for-sama) |

---

## Running the Tests

SAMA includes an automated test suite that runs through GitHub Actions on every push to `main` and on every pull request.

```bash
pip install -e . pytest
pytest            # default suite, about one minute
pytest -m slow    # full SAMA runs with result generation, a few minutes
```

The default suite checks installation, imports, bundled data, input data, the command-line runner, and the convergence of all four optimizers on benchmark functions with known optima (shifted sphere and Rosenbrock), using fixed random seeds. The test settings and pass criteria are documented in [`tests/test_optimizer_convergence.py`](https://github.com/Sas1997/SAMA/blob/main/tests/test_optimizer_convergence.py).

---

## Publications

SAMA has been peer-reviewed and published across multiple journals. If you use SAMA in your research, please cite [A1]:

| # | Reference |
|---|-----------|
| A1 | Sadat, S.A., Takahashi, J. and Pearce, J.M. (2023). A Free and open-source microgrid optimization tool: SAMA. *Energy Conversion and Management*, 298, 117686. [DOI](https://doi.org/10.1016/j.enconman.2023.117686) |
| A2 | Sadat, S.A. and Pearce, J.M. (2024). The threat of economic grid defection in the US with solar PV, battery and generator hybrid systems. *Solar Energy*, 282, 112910. [DOI](https://doi.org/10.1016/j.solener.2024.112910) |
| A3 | Sadat, S.A. and Pearce, J.M. (2025). Techno-economic evaluation of electricity pricing structures on PV and PV-battery hybrid systems in Canada. *Renewable Energy*, 242, 122456. [DOI](https://doi.org/10.1016/j.renene.2025.122456) |
| A4 | Sadat, S.A., Mittal, K. and Pearce, J.M. (2025). Using investments in solar photovoltaics as inflation hedges. *Energies*, 18(4), 890. [DOI](https://doi.org/10.3390/en18040890) |
| A5 | Groza, J.M., Sadat, S.A., Hayibo, K.S. and Pearce, J.M. (2024). Autonomous peer-to-peer virtual net metering of solar PV distributed generation. *Solar Energy Advances*, 4, 100064. [DOI](https://doi.org/10.1016/j.seja.2024.100064) |
| A6 | Sadat, S.A., Lemieux, J.E.B. and Pearce, J.M. (2026). Shifting Subsidies: Redirecting Alberta's Oil and Gas Support to Solar Power. *Energies*, 19(4), 972. [DOI](https://doi.org/10.3390/en19040972) |
| A7 | Sadat, S.A., Hoex, B. and Pearce, J.M. (2022). A review of the effects of haze on solar PV performance. *Renewable and Sustainable Energy Reviews*, 167, 112796. [DOI](https://doi.org/10.1016/j.rser.2022.112796) |

---

## Citation

If you use SAMA in academic work, please cite the primary paper:

```bibtex
@article{sadat2023sama,
  title   = {A Free and open source microgrid optimization tool: {SAMA} the solar alone Multi-Objective Advisor},
  author  = {Sadat, Seyyed Ali and Takahashi, J. and Pearce, Joshua M.},
  journal = {Energy Conversion and Management},
  volume  = {298},
  pages   = {117686},
  year    = {2023},
  doi     = {10.1016/j.enconman.2023.117686}
}
```

---

## License

SAMA is released under the **[GNU Affero General Public License v3.0 (AGPL-3.0)](https://github.com/Sas1997/SAMA/blob/main/LICENSE)**. You are free to use, copy, modify, and distribute SAMA. Any derivative works must also be released under AGPL-3.0, including modified versions that are offered to users over a network.

---

## Contact

**FAST Research Group, Western University, London, Ontario, Canada**

- GitHub: https://github.com/Sas1997/SAMA
- Issues and bug reports: [Open an issue](https://github.com/Sas1997/SAMA/issues)
- Contributing, questions and feature requests: see [CONTRIBUTING.md](https://github.com/Sas1997/SAMA/blob/main/CONTRIBUTING.md)
- For the Windows `.exe` alpha version or reprint requests, contact the developers through the GitHub repository.

---

<p align="center">
  Developed by the <a href="https://github.com/Sas1997/SAMA">FAST Research Group</a> · Western University · AGPL-3.0
</p>


