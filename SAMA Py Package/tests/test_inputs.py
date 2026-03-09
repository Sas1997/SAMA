"""Test 2: Can we load input data?"""

print("="*60)
print("TEST 2: Testing Input Data Loading")
print("="*60)

try:
    from sama.core.Input_Data import InData

    # Check if data loaded
    print(f"\n✓ Load data shape: {InData.Eload.shape}")
    print(f"✓ Solar irradiance shape: {InData.G.shape}")
    print(f"✓ Temperature shape: {InData.T.shape}")
    print(f"✓ Wind speed shape: {InData.Vw.shape}")

    # Check configuration
    print(f"\n✓ PV enabled: {InData.PV == 1}")
    print(f"✓ Wind turbine enabled: {InData.WT == 1}")
    print(f"✓ Battery enabled: {InData.Bat == 1}")
    print(f"✓ Grid connected: {InData.Grid == 1}")
    print(f"✓ EV enabled: {InData.EV == 1}")
    print(f"✓ Heat pump enabled: {InData.HP == 1}")

    # Check optimization parameters
    print(f"\n✓ Population size: {InData.nPop}")
    print(f"✓ Max iterations: {InData.MaxIt}")
    print(f"✓ Number of runs: {InData.Run_Time}")

    print("\n" + "="*60)
    print("✓ INPUT DATA LOADED SUCCESSFULLY!")
    print("="*60)

except Exception as e:
    print(f"\n✗ INPUT LOADING FAILED: {e}")
    import traceback
    traceback.print_exc()
