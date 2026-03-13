"""Test 3: Can we access content files?"""

print("="*60)
print("TEST 3: Testing Content File Access")
print("="*60)

try:
    from samapy import get_content_path
    import os

    # List of files that should be in content folder
    expected_files = [
        'house_load.xlsx',
        'METEO.csv',
        # Add other files you have in content folder
    ]

    print("\nChecking content files:")
    all_found = True

    for filename in expected_files:
        try:
            path = get_content_path(filename)
            exists = os.path.exists(path)

            if exists:
                print(f"✓ {filename} - FOUND at {path}")
            else:
                print(f"✗ {filename} - NOT FOUND (expected at {path})")
                all_found = False
        except Exception as e:
            print(f"✗ {filename} - ERROR: {e}")
            all_found = False

    if all_found:
        print("\n" + "="*60)
        print("✓ ALL CONTENT FILES ACCESSIBLE!")
        print("="*60)
    else:
        print("\n" + "="*60)
        print("✗ SOME CONTENT FILES MISSING!")
        print("="*60)

except Exception as e:
    print(f"\n✗ CONTENT ACCESS FAILED: {e}")
    import traceback
    traceback.print_exc()
