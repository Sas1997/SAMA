"""Test 4: Can we create output directories?"""

print("="*60)
print("TEST 4: Testing Output Directory Creation")
print("="*60)

import os
import shutil

# Clean up any existing test outputs
if os.path.exists('test_sama_outputs'):
    shutil.rmtree('test_sama_outputs')

try:
    import numpy as np
    from sama.results.Results import Gen_Results

    # Create minimal test input
    # [Npv, Nwt, Nbat, N_DG, Cn_I]
    X_test = [10, 0, 50, 1, 25]

    print("\nRunning Gen_Results with test data...")
    print("(This may take 30-60 seconds...)")

    output_path = Gen_Results(X_test, output_dir='test_sama_outputs')

    # Check if directories were created
    if os.path.exists('test_sama_outputs'):
        print(f"\n✓ Output directory created: {os.path.abspath('test_sama_outputs')}")

        # List subdirectories
        for root, dirs, files in os.walk('test_sama_outputs'):
            level = root.replace('test_sama_outputs', '').count(os.sep)
            indent = ' ' * 2 * level
            print(f'{indent}{os.path.basename(root)}/')
            subindent = ' ' * 2 * (level + 1)
            for file in files[:5]:  # Show first 5 files
                print(f'{subindent}{file}')
            if len(files) > 5:
                print(f'{subindent}... and {len(files)-5} more files')

        print("\n" + "="*60)
        print("✓ OUTPUT DIRECTORY TEST PASSED!")
        print("="*60)
    else:
        print("\n✗ Output directory was not created!")

except Exception as e:
    print(f"\n✗ OUTPUT TEST FAILED: {e}")
    import traceback
    traceback.print_exc()
