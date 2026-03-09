# SAMA — Hybrid Renewable Energy System Optimization

SAMA (Solar Alone Multi-objective Advisor) is a Python package for 
optimal design of hybrid renewable energy systems using metaheuristic 
optimization algorithms.

## Supported Components
- Solar PV systems
- Wind turbines
- Battery energy storage (BESS)
- Diesel generators
- Grid connection
- Electric vehicles (EV)
- Heat pump systems (Bosch, Goodman)

## Optimization Algorithms
- **PSO** — Particle Swarm Optimization
- **ADE** — Advanced Differential Evolution
- **ABC** — Artificial Bee Colony
- **GWO** — Grey Wolf Optimizer

## Installation
```bash
pip install sama
```

## Quick Start

### Step 1 — Configure your system
```bash
sama-config
```
This launches an interactive wizard that guides you through all 
parameters and saves a `sama_config_COMPLETE.yaml` file.

### Step 2 — Run the optimization
```bash
sama-run -c sama_config_COMPLETE.yaml
```

### Step 3 — View results
Results are saved to `sama_outputs/` in your working directory.

## Python API
```python
from sama.core.Input_Data import InData
from sama.cli.config_loader import load_config, apply_config

config = load_config("sama_config_COMPLETE.yaml")
apply_config(config)

from sama.optimizers.swarm import Swarm
swarm = Swarm()
swarm.optimize()
```

## Requirements
Python 3.9+ with numpy, pandas, scipy, numba, matplotlib.

## License
GNU General Public License v3.0 — see LICENSE file.