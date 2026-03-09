"""Test 5: Quick optimization test (reduced iterations)"""

print("="*60)
print("TEST 5: Quick Optimization Test")
print("="*60)
print("This will run a SHORT optimization (10 iterations)")
print("to test the complete workflow.")
print("="*60)

import os
import shutil

# Clean up previous test outputs
if os.path.exists('quick_test_outputs'):
    shutil.rmtree('quick_test_outputs')

try:
    from sama.optimizers.swarm import Swarm
    from sama.core.Input_Data import InData

    # Temporarily reduce iterations for quick test
    original_MaxIt = InData.MaxIt
    original_nPop = InData.nPop
    original_RunTime = InData.Run_Time

    InData.MaxIt = 10  # Only 10 iterations
    InData.nPop = 20   # Smaller population
    InData.Run_Time = 1  # Only 1 run

    print(f"\nOriginal settings:")
    print(f"  - Max iterations: {original_MaxIt}")
    print(f"  - Population: {original_nPop}")
    print(f"  - Runs: {original_RunTime}")

    print(f"\nTest settings:")
    print(f"  - Max iterations: {InData.MaxIt}")
    print(f"  - Population: {InData.nPop}")
    print(f"  - Runs: {InData.Run_Time}")

    print("\nStarting optimization...")
    print("="*60)

    # Create optimizer
    optimizer = Swarm()

    # Run optimization
    optimizer.optimize()

    # Restore original settings
    InData.MaxIt = original_MaxIt
    InData.nPop = original_nPop
    InData.Run_Time = original_RunTime

    print("\n" + "="*60)
    print("✓ QUICK OPTIMIZATION TEST PASSED!")
    print("="*60)
    print("\nCheck the output folders for results:")
    print("  - sama_inputs/")
    print("  - sama_outputs/")

except Exception as e:
    print(f"\n✗ OPTIMIZATION TEST FAILED: {e}")
    import traceback
    traceback.print_exc()

    # Restore settings even on error
    try:
        InData.MaxIt = original_MaxIt
        InData.nPop = original_nPop
        InData.Run_Time = original_RunTime
    except:
        pass
