"""
SAMAPy CLI Package
================
Provides the interactive wizard (samapy-config) and the optimization
runner (samapy-run) for SAMAPy hybrid energy system optimization.

Public API
----------
    load_config(path)           Load a YAML config file → dict
    apply_config(config)        Apply config dict to InData singleton → InData
    merge_configs(*paths)       Merge multiple YAML files → dict
    save_config(config, path)   Save config dict to YAML file

Entry points (defined in setup.py)
-----------------------------------
    samapy-config  →  samapy.cli.wizard:main
    samapy-run     →  samapy.cli.runner:main
"""

from .config_loader import load_config, apply_config, merge_configs, save_config

__version__ = '1.0.0'
__all__ = [
    'load_config',
    'apply_config',
    'merge_configs',
    'save_config',
]
