"""Test 1: Can we import SAMAPy?"""

print("=" * 60)
print("TEST 1: Testing SAMAPy imports")
print("=" * 60)

try:
    # Test core imports
    from samapy.core.Input_Data import InData

    print("✓ Input_Data imported successfully")

    # Test optimizer imports
    from samapy.optimizers.swarm import Swarm

    print("✓ Swarm optimizer imported successfully")

    # Test results imports
    from samapy.results.Results import Gen_Results

    print("✓ Results module imported successfully")

    # Test EMS imports
    from samapy.ems.EMS import EMS

    print("✓ EMS module imported successfully")

    print("\n" + "=" * 60)
    print("✓ ALL IMPORTS SUCCESSFUL!")
    print("=" * 60)

except Exception as e:
    print(f"\n✗ IMPORT FAILED: {e}")