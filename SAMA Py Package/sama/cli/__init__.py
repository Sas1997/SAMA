"""
SAMA CLI Package
================
Provides the interactive wizard (sama-config) and the optimization
runner (sama-run) for SAMA hybrid energy system optimization.

Public API
----------
    load_config(path)           Load a YAML config file → dict
    apply_config(config)        Apply config dict to InData singleton → InData
    merge_configs(*paths)       Merge multiple YAML files → dict
    save_config(config, path)   Save config dict to YAML file

Entry points (defined in setup.py)
-----------------------------------
    sama-config  →  sama.cli.wizard:main
    sama-run     →  sama.cli.runner:main
"""

from .config_loader import load_config, apply_config, merge_configs, save_config

__version__ = '2.0.0'
__all__ = [
    'load_config',
    'apply_config',
    'merge_configs',
    'save_config',
]
