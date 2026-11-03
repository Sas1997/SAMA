# SAMA examples and run routes

This folder holds the reference case and an example script. The table below shows, for each way of running SAMA, where the inputs come from, which configuration is used, the command, and where the outputs go.

## Reference case

`reference_config.yaml` is the reference run documented in [Section 18 of the documentation](../Docs/SAMA_Documentation.md#18-sample-optimization-output): a residential property in Boston, MA, with PV, a Bosch heat pump and an EV under a time-of-use tariff, optimized with ADE. Run it from the repository root:

```bash
samapy-run -c examples/reference_config.yaml --seed 1 --no-gui --output reference_run
# or, from a checkout without installing the package:
python run_samapy_optimized.py -c examples/reference_config.yaml --seed 1 --no-gui --output reference_run
```

With `--seed 1` the results should match Section 18 to the cent. Section 18.7 explains how to compare results from other seeds.

## Run routes

| Route | Inputs | Configuration | Command | Outputs |
|---|---|---|---|---|
| **1. Reference case** | SAMA's example data in `src/samapy/content/` (see below) | `examples/reference_config.yaml` | `samapy-run -c examples/reference_config.yaml --seed 1 --no-gui --output reference_run` | `reference_run/` (figures in `figs/`, data in `data/`) |
| **2. Your own case with the wizard** | Your files in the input folder you choose in the wizard, or SAMA's example data | `samapy_config_COMPLETE.yaml`, written by `samapy-config` in the folder you run it from | `samapy-config`, then `samapy-run` in the same folder | `samapy_outputs/` (or the folder chosen in the wizard) |
| **3. Repository checkout** | as in route 1 or 2 | any YAML file; default `samapy_config_COMPLETE.yaml` in the current folder | `python run_samapy_optimized.py -c <file>` (same options as `samapy-run`) | as in route 1 or 2 |
| **4. Python script** | SAMA's example data, paths set in the script | the `Config` class in `examples/optimization_case.py` | `python examples/optimization_case.py` | `samapy_outputs/` in the current folder |
| **5. Defaults in `Input_Data.py`** | the files named in `src/samapy/core/Input_Data.py` | none; edit `Input_Data.py` | `python src/samapy/optimizers/pso.py` (or `ade.py`, `run_abc.py`, `gwo.py`) | `samapy_outputs/` in the current folder |

Routes 1 to 3 read a YAML configuration: the values in the file are applied and all derived quantities are recalculated, and parameters that are not in the file keep their defaults from `Input_Data.py`. `samapy-run -c <file> --dry-run --verbose` checks a configuration without running it and lists the input files it will read. Add `--mode multi` to any YAML route for the multi-objective mode (Pareto front in `pareto_front.csv` and `pareto_front.png`).

## SAMA's example data (`src/samapy/content/`)

| File | Contents | Used by the reference case |
|---|---|---|
| `METEO.csv` | NSRDB weather file for Boston, MA (irradiance, temperature, wind speed, pressure) | yes |
| `house_load.xlsx` | Hourly heating and cooling loads of the example house | yes |
| `Generic_load_JulyP.csv`, `Generic_load_JanuaryP.csv` | Generic hourly load shapes with a July or a January peak (load types 5, 8 and 9) | July shape, scaled to 9,500 kWh/yr |
| `HP_Bosch/`, `HP_Goodman/` | Manufacturer heat pump performance tables | Bosch |
| `Eload.csv` | Example hourly electrical load (load type 1) | no |
| `Eload_daily.csv` | Example daily load profile (load type 10) | no |
| `Irradiance.csv`, `Temperature.csv`, `WSPEED.csv` | Example hourly plane-of-array irradiance, temperature and wind speed (input method 2 of each) | no |

To use your own data, run `samapy-config` and give the folder with your files, or set the file paths in the configuration (documentation, Section 4.6). The YAML routes (1 to 3) never change the files in `content/`. The Python script of route 4 copies the files named in its `Config` class into `content/`, because `Input_Data.py` reads its inputs from there; with its default paths nothing is copied.

## Optimizer files (`src/samapy/optimizers/`)

| File | Role |
|---|---|
| `swarm.py`, `AdvancedDifferentialEvolution.py`, `ArtificialBeeColony.py`, `GreyWolfOptimizer.py` | Algorithm modules with the optimizer classes (`Swarm`, `AdvancedDifferentialEvolution`, `ImprovedArtificialBeeColony`, `GreyWolfOptimizer`) |
| `multiobjective.py` | Multi-objective mode (NSGA-II from pymoo) |
| `_common.py` | Settings, bounds and random-seed helpers shared by all optimizers |
| `pso.py`, `ade.py`, `run_abc.py`, `gwo.py` | Launcher scripts for route 5. `run_abc.py` is not called `abc.py` because that name would hide Python's standard `abc` module when the script is run from its folder |

The file names are kept so that existing scripts and imports continue to work.
