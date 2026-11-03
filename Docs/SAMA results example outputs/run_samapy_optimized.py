#!/usr/bin/env python3
"""
Run SAMA from a repository checkout with a YAML configuration.

This script is the same as the `samapy-run` command installed by
`pip install samapy` (or `pip install -e .`). It applies the configuration
with the full set of derived recalculations (samapy.cli.config_loader) and
runs the optimization.

Usage, from the repository root:
    python run_samapy_optimized.py                                  # samapy_config_COMPLETE.yaml
    python run_samapy_optimized.py -c my_config.yaml -a pso
    python run_samapy_optimized.py --mode multi --no-gui
    python run_samapy_optimized.py --seed 1 --no-gui --output reference_run

Run `python run_samapy_optimized.py --help` for all options.
"""
import sys
from pathlib import Path

# Use the package in src/ when run from a checkout without installing it
_SRC = Path(__file__).resolve().parent / "src"
if (_SRC / "samapy").is_dir() and str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from samapy.cli.runner import main  # noqa: E402

if __name__ == "__main__":
    sys.exit(main())
