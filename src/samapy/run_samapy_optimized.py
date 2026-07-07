#!/usr/bin/env python3
"""
SAMAPy Optimization Runner - Best Approach

This uses SAMAPy's singleton pattern without modifying Input_Data.py file.
It modifies the InData instance at runtime before optimization starts.

Usage:
    python run_samapy_optimized.py --algorithm pso
"""

import yaml
from pathlib import Path
import sys
import argparse
import numpy as np

def load_config(config_path):
    """Load YAML configuration"""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config if config else {}

def apply_config_to_indata(config):
    """
    Apply configuration to InData singleton
    This must be done AFTER importing InData but BEFORE running optimizer
    """
    from samapy.core.Input_Data import InData

    applied = 0
    skipped = []

    for key, value in config.items():
        if hasattr(InData, key):
            # Handle numpy array conversion
            original = getattr(InData, key)
            if isinstance(original, np.ndarray) and isinstance(value, list):
                value = np.array(value)

            # Set the value
            setattr(InData, key, value)
            applied += 1
        else:
            skipped.append(key)

    print(f"✅ Applied {applied} parameters to InData")
    if skipped and len(skipped) <= 10:
        print(f"   Skipped {len(skipped)} non-InData parameters: {', '.join(skipped[:5])}")
    elif skipped:
        print(f"   Skipped {len(skipped)} metadata parameters")

    return applied

def run_algorithm(algorithm):
    """Import and run the specified optimization algorithm"""
    print(f"\n{'='*80}")
    print(f"RUNNING {algorithm.upper()} OPTIMIZATION")
    print(f"{'='*80}\n")

    try:
        if algorithm == 'pso':
            from samapy.optimizers.swarm import Swarm
            from samapy.core.Input_Data import InData
            print(f"Configuration check:")
            print(f"  MaxIt: {InData.MaxIt}")
            print(f"  nPop: {InData.nPop}")
            print(f"  Run_Time: {InData.Run_Time}\n")

            swarm = Swarm()
            swarm.optimize()

        elif algorithm == 'ade':
            from samapy.optimizers.AdvancedDifferentialEvolution import AdvancedDifferentialEvolution
            ade = AdvancedDifferentialEvolution()
            ade.optimize()

        elif algorithm == 'gwo':
            from samapy.optimizers.GreyWolfOptimizer import GreyWolfOptimizer
            gwo = GreyWolfOptimizer()
            gwo.optimize()

        elif algorithm == 'abc':
            from samapy.optimizers.ArtificialBeeColony import ImprovedArtificialBeeColony
            abc = ImprovedArtificialBeeColony()
            abc.optimize()
        else:
            print(f"❌ Unknown algorithm: {algorithm}")
            return False

        return True

    except Exception as e:
        print(f"❌ Error during optimization: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    parser = argparse.ArgumentParser(
        description='Run SAMAPy optimization with wizard-configured parameters'
    )
    parser.add_argument(
        '--algorithm', '-a',
        type=str,
        required=True,
        choices=['pso', 'ade', 'gwo', 'abc'],
        help='Optimization algorithm: pso, ade, gwo, or abc'
    )
    parser.add_argument(
        '--config', '-c',
        type=str,
        default='samapy_config_COMPLETE_HYBRID.yaml',
        help='Path to configuration YAML file (default: samapy_config_COMPLETE_HYBRID.yaml)'
    )

    args = parser.parse_args()

    print("""
╔═══════════════════════════════════════════════════════════════════════════╗
║         SAMAPy Optimization with Wizard Configuration (Runtime)           ║
╚═══════════════════════════════════════════════════════════════════════════╝
""")

    # Check config file exists
    config_path = Path(args.config)
    if not config_path.exists():
        print(f"❌ Configuration file not found: {config_path}")
        print("\nPlease run the wizards first to create configuration:")
        print("  python samapy/samapy/cli/samapy_wizard_1_load_weather.py")
        print("  python samapy/samapy/cli/samapy_wizard_2_rate_structures.py")
        print("  python samapy/samapy/cli/samapy_wizard_3_MASTER.py")
        sys.exit(1)

    try:
        # Step 1: Load configuration
        print(f"Step 1: Loading configuration from {config_path}...")
        config = load_config(config_path)
        print(f"  Loaded {len(config)} parameters\n")

        # Step 2: Apply to InData (MUST be done before importing optimizers)
        print("Step 2: Applying configuration to InData singleton...")
        apply_config_to_indata(config)

        # Step 3: Run optimization
        print(f"\nStep 3: Starting {args.algorithm.upper()} optimization...")
        success = run_algorithm(args.algorithm)

        if success:
            print("\n" + "="*80)
            print("✅ OPTIMIZATION COMPLETED SUCCESSFULLY!")
            print("="*80)
            print("\nResults saved in: samapy_outputs/")
            print("  - Optimization.png (convergence curve)")
            print("  - Results files")
        else:
            print("\n" + "="*80)
            print("❌ OPTIMIZATION FAILED")
            print("="*80)
            sys.exit(1)

    except KeyboardInterrupt:
        print("\n\n❌ Cancelled by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Fatal error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == '__main__':
    main()
