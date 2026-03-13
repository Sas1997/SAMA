"""Test 6: Full optimization with actual settings"""

print("="*60)
print("TEST 6: Full Optimization Test")
print("="*60)
print("⚠️  WARNING: This will run a FULL optimization!")
print("   This may take 10-30 minutes depending on your computer.")
print("="*60)

import time

# Ask for confirmation
response = input("\nDo you want to proceed? (yes/no): ")

if response.lower() != 'yes':
    print("Test cancelled.")
    exit()

try:
    from samapy.optimizers.swarm import Swarm
    from samapy.core.Input_Data import InData

    print(f"\nOptimization settings:")
    print(f"  - Max iterations: {InData.MaxIt}")
    print(f"  - Population size: {InData.nPop}")
    print(f"  - Number of runs: {InData.Run_Time}")
    print(f"  - System components:")
    print(f"    PV: {InData.PV == 1}")
    print(f"    Wind: {InData.WT == 1}")
    print(f"    Battery: {InData.Bat == 1}")
    print(f"    Generator: {InData.DG == 1}")
    print(f"    Grid: {InData.Grid == 1}")
    print(f"    EV: {InData.EV == 1}")
    print(f"    Heat Pump: {InData.HP == 1}")

    print("\n" + "="*60)
    print("Starting full optimization...")
    print("="*60 + "\n")

    start_time = time.time()

    # Create and run optimizer
    optimizer = Swarm()
    optimizer.optimize()

    end_time = time.time()
    elapsed_time = end_time - start_time

    print("\n" + "="*60)
    print("✓ FULL OPTIMIZATION COMPLETED!")
    print("="*60)
    print(f"\nTotal execution time: {elapsed_time:.2f} seconds ({elapsed_time/60:.2f} minutes)")
    print("\nResults saved to:")
    print("  - samapy_inputs/")
    print("  - samapy_outputs/")

except Exception as e:
    print(f"\n✗ OPTIMIZATION FAILED: {e}")
    import traceback
    traceback.print_exc()
