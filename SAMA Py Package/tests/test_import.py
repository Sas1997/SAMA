"""Test 1: Can we import SAMA?"""

print("=" * 60)
print("TEST 1: Testing SAMA imports")
print("=" * 60)

try:
    # Test core imports
    from sama.core.Input_Data import InData

    print("✓ Input_Data imported successfully")

    # Test optimizer imports
    from sama.optimizers.swarm import Swarm

    print("✓ Swarm optimizer imported successfully")

    # Test results imports
    from sama.results.Results import Gen_Results

    print("✓ Results module imported successfully")

    # Test EMS imports
    from sama.ems.EMS import EMS

    print("✓ EMS module imported successfully")

    print("\n" + "=" * 60)
    print("✓ ALL IMPORTS SUCCESSFUL!")
    print("=" * 60)

except Exception as e:
    print(f"\n✗ IMPORT FAILED: {e}")