<p align="center">
  <img src="Assets/SAMA_Logo-with_Typography.png" alt="SAMA Logo" width="420"/>
</p>

<h2 align="center">Solar Alone Multi-Objective Advisor</h2>
<h4 align="center">Open-Source Hybrid Renewable Energy Microgrid Optimization</h4>

<p align="center">
  <a href="https://doi.org/10.1016/j.enconman.2023.117686"><img src="https://img.shields.io/badge/DOI-10.1016%2Fj.enconman.2023.117686-blue?style=flat-square" alt="DOI"/></a>
  <a href="https://github.com/Sas1997/SAMA/blob/main/LICENSE"><img src="https://img.shields.io/badge/License-GPL%20v3-green?style=flat-square" alt="License: GPL v3"/></a>
  <a href="https://pypi.org/project/SAMAbyRenXera/"><img src="https://img.shields.io/pypi/v/SAMAbyRenXera?style=flat-square&label=PyPI&color=orange" alt="PyPI"/></a>
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
- [Optimization Algorithms](#optimization-algorithms)
- [Output Examples](#output-examples)
- [Documentation](#documentation)
- [Publications](#publications)
- [Citation](#citation)
- [License](#license)
- [Contact](#contact)

---

## Why SAMA?

Proprietary tools like HOMER Pro and HOMER Grid are closed-source and license-restricted. SAMA was built to remove that barrier:

- **Free and open-source**: GPL-3.0, forever
- **Transparent algorithms**: read, modify, and extend every line
- **Validated**: cross-validated against HOMER Pro across multiple U.S. climate zones
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

Constraints include reliability (LPSP), renewable energy fraction, capital budget, and rooftop area limits. Multi-objective mode (`EM=1`) adds **Levelized Emissions (LEM)** as a second objective.

---

## Key Capabilities

| Category | Details |
|----------|---------|
| **System Components** | Solar PV, wind turbines, battery storage (Li-ion or lead-acid), diesel generator, utility grid, air-source heat pump (Bosch or Goodman), electric vehicle with V2X |
| **Optimization Algorithms** | Particle Swarm Optimization (PSO), Advanced Differential Evolution (ADE), Artificial Bee Colony (ABC), Grey Wolf Optimizer (GWO) |
| **Energy Simulation** | 8,760-hour hourly dispatch with marginal-cost-based EMS, EV smart charging and V2X lookahead arbitrage, heat pump thermal-electric coupling |
| **Electricity Tariffs** | 8 structures: flat, seasonal, monthly, tiered, seasonal tiered, monthly tiered, Time-of-Use (TOU), Ultra-Low TOU (ULO) |
| **Natural Gas Tariffs** | 8 structures including PG&E G-1 therms-based and Enbridge EGD 4-tier m³-based |
| **Financial Analysis** | 25-year NPC, LCOE, levelized emissions, price escalation, RE incentives (30% ITC), net metering reconciliation |
| **Load Inputs** | 10 electrical load input modes (CSV, monthly/annual averages, generic profiles), thermal load from Excel |
| **Weather Inputs** | SAM NSRDB API integration, user CSV files, monthly/annual averages |
| **Outputs** | Convergence plots, 25-year cash flow charts, energy distribution, battery SOC, grid interconnection, EV and heat pump performance charts, hourly CSV data |

---

## Getting Started

SAMA supports three usage methods. Choose the one that fits your workflow:

### Method 1: Raw Python Code

For researchers and developers who want full access to source code and algorithms.

```bash
git clone https://github.com/Sas1997/SAMA.git
cd "SAMA/Backend Codes/SAMA V2.0.1-GitHub"
pip install numpy pandas scipy numba matplotlib openpyxl seaborn numpy-financial questionary PyYAML
```

Edit `sama/core/Input_Data.py` to configure your system, then run:

```bash
python run_sama_optimized.py --algorithm pso
```

Available algorithms: `pso`, `ade`, `abc`, `gwo`

### Method 2: Python Package (pip)

The recommended method for most users. No source editing required.

```bash
pip install sama
```

> **Note:** The PyPI distribution name is `SAMAbyRenXera`. Once installed, import and CLI commands use `sama`.

**Step 1: Configure your project:**
```bash
sama-config
```
An interactive 18-section wizard guides you through all parameters and saves `sama_config_COMPLETE.yaml`.

**Step 2: Run the optimization:**
```bash
sama-run
```

**Step 3: View results in `sama_outputs/`**

Additional run options:

```bash
sama-run --algorithm ade          # override algorithm
sama-run --output results/run1    # override output directory
sama-run --dry-run --verbose      # validate config without running
sama-run --no-gui                 # headless / server mode
```

### Method 3: Windows .exe (Alpha)

A point-and-click GUI for non-technical users. No Python installation required. Currently in alpha, contact the developers to request access.

> https://github.com/Sas1997/SAMA

---

## Quick Start

Run SAMA in five steps using bundled sample data (London, Ontario):

```bash
# 1. Install
pip install sama

# 2. Create a project folder and navigate into it
mkdir my_sama_project && cd my_sama_project

# 3. Launch the configuration wizard (accept defaults for a quick test)
sama-config

# 4. Run the optimization
sama-run

# 5. View your results
#    sama_outputs/figs/    → all charts (PNG/SVG)
#    sama_outputs/data/    → Outputforplotting.csv (8,760-hour time series)
```

> **Expected runtime:** 5–30 minutes for `MaxIt=200`, `nPop=50` depending on hardware. The first run is slower due to one-time Numba JIT compilation of the EMS engine.

---

## Optimization Algorithms

All four algorithms optimize the same 5-variable design space. Choose based on your preference for convergence speed vs. exploration depth:

| Algorithm | Key Behavior | Best For |
|-----------|-------------|----------|
| **PSO**: Particle Swarm Optimization | Velocity-based swarm convergence | Fast convergence |
| **ADE**: Advanced Differential Evolution | Multi-strategy mutation with adaptive F and CR | Thorough exploration |
| **ABC**: Artificial Bee Colony | Foraging phases with guided reinitialization | Balanced exploration |
| **GWO**: Grey Wolf Optimizer | Pack hierarchy convergence; parallel variant available | Fast convergence, parallel runs |

For rigorous comparison, set `Run_Time > 1` to average across multiple independent runs.

---

## Output Examples

After `sama-run` completes, results are organized in `sama_outputs/`:

```
sama_outputs/
├── Optimization.png                  ← convergence curve
├── figs/
│   ├── Cash_Flow.svg                 ← 25-year lifecycle cost breakdown
│   ├── Energy Distribution.png       ← annual energy supply by source
│   ├── Battery State of Charge.png   ← hourly SOC profile
│   ├── Grid Interconnection.png      ← hourly grid import/export
│   ├── electricity_comparison.png    ← SAMA system vs. grid-only cost
│   ├── hp_monthly_summary.png        ← heat pump COP and energy (if HP=1)
│   └── EV Energy.png                 ← EV charge/discharge (if EV=1)
└── data/
    ├── Outputforplotting.csv         ← full 8,760-hour time series
    └── cash_flow_data.csv            ← annual 25-year cash flow table
```

---

## Documentation

Full documentation is available in this repository:

- **[SAMA_Documentation_v3_final.md](./Docs/SAMA_Documentation_v3_final.md)**: Complete user guide covering all parameters, algorithms, financial models, input formats, troubleshooting, and mathematical formulations

Key documentation sections:

| Topic | Location |
|-------|----------|
| All configuration parameters | [Section 6: Configuration File](./Docs/SAMA_Documentation_v3_final.md#6-the-configuration-file-sama_config_completeyaml) |
| Optimization algorithms (detail) | [Section 7](./Docs/SAMA_Documentation_v3_final.md#7-optimization-algorithms) |
| Energy management system | [Section 8](./Docs/SAMA_Documentation_v3_final.md#8-energy-management-system-ems) |
| Component models | [Section 9](./Docs/SAMA_Documentation_v3_final.md#9-component-models) |
| Financial model | [Section 10](./Docs/SAMA_Documentation_v3_final.md#10-financial-model) |
| Mathematical formulations | [Section 17](./Docs/SAMA_Documentation_v3_final.md#17-technical-framework-and-mathematical-formulation) |
| Where to find input data | [Appendix A](./Docs/SAMA_Documentation_v3_final.md#appendix-a-where-to-find-input-data-for-sama) |
| Troubleshooting | [Section 13](./Docs/SAMA_Documentation_v3_final.md#13-troubleshooting) |

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

SAMA is released under the **[GNU General Public License v3.0 (GPL-3.0)](./LICENSE)**. You are free to use, copy, modify, and distribute SAMA. Any derivative works must also be released under GPL-3.0.

---

## Contact

**FAST Research Group, Western University, London, Ontario, Canada**

- GitHub: https://github.com/Sas1997/SAMA
- Issues and bug reports: [Open an issue](https://github.com/Sas1997/SAMA/issues)
- For the Windows `.exe` alpha version or reprint requests, contact the developers through the GitHub repository.

---

<p align="center">
  Made with ☀️ by the <a href="https://github.com/Sas1997/SAMA">FAST Research Group</a> · Western University · GPL-3.0
</p>
