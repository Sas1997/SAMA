# SAMAPy: Solar Alone Multi-Objective Advisor Python

<h1 align="center">
  <img src="https://github.com/Sas1997/SAMA/raw/main/Assets/SAMA_Logo-with_Typography.png" width="400" alt="SAMAPy Logo"/>
</h1>

<h3 align="center">Solar Alone Multi-Objective Advisor - Python Package</h3>

<p align="center">
  <em>Free and open-source hybrid renewable energy system optimization</em><br>
  <em>FAST Research Group, Western University, London, Ontario, Canada</em>
</p>

<p align="center">
  <img src="https://img.shields.io/pypi/v/samapy?color=blue&label=PyPI" alt="PyPI version"/>
  <img src="https://img.shields.io/pypi/pyversions/samapy" alt="Python versions"/>
  <img src="https://img.shields.io/badge/license-GPLv3-green" alt="License"/>
  <img src="https://img.shields.io/badge/platform-Windows%20%7C%20macOS%20%7C%20Linux-lightgrey" alt="Platform"/>
  <img src="https://img.shields.io/badge/DOI-10.1016%2Fj.enconman.2023.117686-orange" alt="DOI"/>
</p>

---

**SAMAPy** is an open-source Python package for the optimal sizing and techno-economic analysis of hybrid renewable energy systems (HRES). Developed at the **FAST Research Group, Western University**, it combines physics-based energy simulation with state-of-the-art metaheuristic optimization to find the lowest-cost system configuration that satisfies reliability and renewable energy constraints.

SAMAPy handles the full modelling stack (solar PV, wind turbines, battery storage, diesel generators, grid interconnection, heat pumps, and electric vehicles) under real-world utility pricing structures including Time-of-Use, tiered rates, and Ontario's Ultra-Low Overnight tariff.

---

## Table of Contents

- [Key Features](#key-features)
- [Supported Components](#supported-components)
- [Optimization Algorithms](#optimization-algorithms)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [CLI Workflow](#cli-workflow)
- [Python API](#python-api)
- [Input Data](#input-data)
- [Outputs](#outputs)
- [Examples](#examples)
- [Contributing](#contributing)
- [License](#license)
- [Citation](#citation)

---

## Key Features

- **Multi-component modelling**: PV, wind, battery (Li-ion & lead-acid), diesel generator, grid, heat pump (Bosch & Goodman), and EV with V2G
- **8 electricity rate structures**: flat, seasonal, monthly, tiered, seasonal-tiered, monthly-tiered, TOU, and ultra-low overnight (ULO)
- **8 natural gas rate structures**: supporting $/m³, $/kWh, and $/therm pricing with automatic unit conversion
- **4 metaheuristic optimizers**: PSO, ADE, GWO, and ABC, each tunable from a YAML config
- **Comprehensive economic analysis**: NPC, LCOE, IRR, payback period, cash flow charts, avoided-cost breakdown
- **Net Energy Metering** (NEM): annual credit reconciliation with configurable caps
- **Rich result outputs**: 15+ publication-quality figures, hourly CSV data, and printed performance tables
- **Interactive configuration wizard**: `samapy-config` guides you through every parameter and writes a ready-to-run YAML file
- **One-command optimization**: `samapy-run` reads your YAML and launches the chosen algorithm

---

## Supported Components

| Component | Models / Options |
|---|---|
| Solar PV | Monofacial, configurable tilt & azimuth, SAM-based POA irradiance |
| Wind Turbine | Generic power curve, hub-height wind-shear correction |
| Battery | Li-ion (default), Lead-Acid |
| Diesel Generator | Generic linear fuel curve |
| Grid | Buy/sell with NEM, configurable import/export limits |
| Heat Pump | Bosch (BOVA series), Goodman — full COP lookup tables |
| Electric Vehicle | V2G capable, travel-pattern modelling, degradation tracking |

---

## Optimization Algorithms

| Algorithm | Flag | Strengths |
|---|---|---|
| Particle Swarm Optimization | `pso` | Fast convergence, well-tuned defaults |
| Advanced Differential Evolution | `ade` | Self-adaptive, good for multimodal problems |
| Grey Wolf Optimizer | `gwo` | Strong global search |
| Artificial Bee Colony | `abc` | Balanced exploration / exploitation |

---

## Installation

**Requires Python 3.9 or later.**

```bash
## Installation

### Stable release (Not Released Yet!)
pip install samapy

### Testing release (for testers and early adopters)
To install the latest test version from TestPyPI:

pip install -i https://test.pypi.org/simple/ --extra-index-url https://pypi.org/simple/ samapy

```

To install the latest development version directly from source:

```bash
git clone https://github.com/Sas1997/SAMA.git
cd SAMA
pip install -e .
```

Verify the installation:

```bash
python -c "import samapy; print('SAMAPy installed successfully')"
```

---

## Quick Start

The fastest way to run SAMAPy is through its two CLI commands.

### Step 1: Generate a configuration file

```bash
samapy-config
```

This launches an interactive wizard that walks you through every system parameter — components, pricing, economics, optimizer settings — and writes a `samapy_config_COMPLETE.yaml` file in your working directory.

### Step 2: Run the optimization

```bash
samapy-run
```

SAMAPy reads your YAML, runs the chosen algorithm, and saves all results to `samapy_outputs/`.

### Step 3: Inspect results

```
samapy_outputs/
├── figs/
│   ├── Cash_Flow.svg
│   ├── Energy Distribution.png
│   ├── Battery State of Charge.png
│   ├── Grid Hourly Cost.png
│   ├── hp_heating_performance.png
│   ├── hp_cop_vs_temp.png
│   └── ...
└── data/
    └── Outputforplotting.csv
```

---

## CLI Workflow

### `samapy-config`

```
samapy-config
```

Launches the step-by-step configuration wizard. No arguments required. Produces `samapy_config_COMPLETE.yaml`.

### `samapy-run`

```
samapy-run [OPTIONS]

Options:
  -c, --config PATH      Path to YAML config  [default: samapy_config_COMPLETE.yaml]
  -a, --algorithm ALGO   Override algorithm: pso | ade | gwo | abc
  --output DIR           Override output directory  [default: samapy_outputs]
  --dry-run              Validate config without running
  --no-gui               Disable matplotlib GUI (for headless servers)
  --verbose              Show full parameter load log and config summary
```

**Examples:**

```bash
# Use defaults
samapy-run

# Explicit config path and algorithm
samapy-run -c my_project/config.yaml -a ade

# Validate config without running optimization
samapy-run --dry-run --verbose

# Run on a headless server
samapy-run --no-gui --output /results/run_01
```

---

## Python API

For programmatic use, scripting, and integration into research workflows.

### Minimal example

```python
from samapy.cli.config_loader import load_config, apply_config
from samapy.optimizers.swarm import Swarm

# Load a wizard-generated YAML
config = load_config("samapy_config_COMPLETE.yaml")
indata = apply_config(config)

# Run PSO optimization
optimizer = Swarm()
optimizer.optimize()
```

### Accessing content data files

SAMAPy bundles default weather and load data. Use `get_content_path` to locate them reliably regardless of where the package is installed:

```python
from samapy import get_content_path

meteo_path  = get_content_path("METEO.csv")
eload_path  = get_content_path("Eload.csv")
house_path  = get_content_path("house_load.xlsx")
```

### Accessing input data

After `apply_config` (or directly after `import`), all simulation parameters live on the `InData` singleton:

```python
from samapy.core.Input_Data import InData

print(f"Annual load:     {InData.Eload.sum():.0f} kWh")
print(f"Peak irradiance: {InData.G.max():.0f} W/m²")
print(f"Min temperature: {InData.T.min():.1f} °C")
print(f"Discount rate:   {InData.ir*100:.2f}%")
```

### Generating results manually

```python
from samapy.results.Results import Gen_Results
import numpy as np

# X = [N_PV, N_WT, N_bat, N_DG, Cn_inverter]
X = np.array([12.0, 0.0, 8.0, 0.0, 10.5])
Gen_Results(X, output_dir="my_outputs")
```

---

## Input Data

### Required files

| File | Description | Format |
|---|---|---|
| `Eload.csv` | Hourly electrical load | 8760 rows, 1 column, kWh |
| `METEO.csv` | Hourly weather data | NSRDB/SAM format with header |
| `house_load.xlsx` | Hourly heating & cooling loads | Column 1: heating kWh, Column 2: cooling kWh |

Place your files in the working directory and point to them via the wizard or YAML config. SAMAPy copies them into the package content folder automatically.

### Built-in defaults

SAMAPy ships with London, Ontario TMY data so you can run a test optimization immediately after installation without providing any files.

### Weather data

Weather data must follow the NSRDB CSV format (as exported from [https://nsrdb.nrel.gov](https://nsrdb.nrel.gov)). SAMAPy uses NREL's SAM libraries internally to compute plane-of-array (POA) irradiance from the global horizontal values in the file.

---

## Outputs

### Console output

SAMAPy prints a full techno-economic summary after every optimization run:

```
------------- System Size --------------
Cpv  (kW) = 12.0
Cbat (kWh) = 8.004
Cdg  (kW) = 0.0
Cinverter (kW) = 10.5

************* Economic Results *************
NPC  = $ 48,321.44
LCOE = 0.14 $/kWh
IRR  = 8.73%
Payback Period: 9 years

============ Technical Results =============
PV Power  = 18,432 kWh
RE  = 94.32 %
LPSP Total = 0.00 %
```

### Figures saved to `samapy_outputs/figs/`

| File | Description |
|---|---|
| `Cash_Flow.svg` | Annual cash flow waterfall with cumulative total |
| `Cash Flow_ADV.png` | Cash flow vs. grid-only baseline |
| `Energy Distribution.png` | Hourly power dispatch over the year |
| `Battery State of Charge.png` | Battery SOC profile |
| `Grid Interconnection.png` | Hourly grid import and export |
| `Grid Hourly Cost.png` | Colour map of hourly electricity price |
| `Specific day results.png` | Detailed 24-hour dispatch for a sample day |
| `electricity_comparison.png` | Bill comparison: with vs. without system |
| `hp_heating_performance.png` | HP heating load vs. electrical power consumed |
| `hp_cooling_performance.png` | HP cooling load vs. electrical power consumed |
| `hp_cop_vs_temp.png` | COP as a function of ambient temperature |
| `hp_monthly_summary.png` | Monthly energy, power, and COP bar charts |
| `EV Energy.png` | EV SOC and charge/discharge profile |

### Data saved to `samapy_outputs/data/`

| File | Description |
|---|---|
| `Outputforplotting.csv` | Full 8760-row hourly dataset of all power flows, prices, SOC, and environmental variables |

---

## Examples

See the **[Examples](https://github.com/Sas1997/SAMA/blob/main/Docs/SAMA_Documentation.md#17-examples)** and **[Sample Optimization Output](https://github.com/Sas1997/SAMA/blob/main/Docs/SAMA_Documentation.md#18-sample-optimization-output)** sections of the documentation for five complete worked examples covering:

1. **Basic solar + battery + grid**: the simplest grid-tied PV system
2. **Full hybrid system**: PV + wind + battery + diesel generator
3. **Heat pump integration**: adding a Goodman or Bosch heat pump
4. **Electric vehicle (V2X)**: EV with vehicle-to-grid discharge
5. **YAML-driven workflow**: using the wizard output programmatically

---

## Contributing

Contributions are welcome. Please open an issue to discuss your idea before submitting a pull request.

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/your-feature`)
3. Commit your changes with clear messages
4. Push and open a pull request against `main`

Please follow PEP 8 style and include docstrings for any new public functions.

---

## License

SAMAPy is distributed under the **GNU General Public License v3.0**. See [`License`](License) for the full text.

---

## Citation

If you use SAMAPy in your research, please cite:

```bibtex
@software{sadat2024samapy,
  author    = {Sadat, Seyyed Ali},
  title     = {{SAMAPy}: Solar Alone Multi-Objective Advisor — Python Package},
  year      = {2024},
  url       = {https://github.com/Sas1997/SAMA},
  note      = {FAST Research Group, Western University, London, Ontario, Canada}
}
```

---

## Contact

**Seyyed Ali Sadat**, FAST Research Group, Western University
📧 alisadat942@gmail.com
🔗 [github.com/Sas1997/SAMA](https://github.com/Sas1997/SAMA)
