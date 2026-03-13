#!/usr/bin/env python3

"""
SAMAPy Configuration Wizard
Runs samapy_Wizard.py by importing it directly
"""

import sys
from pathlib import Path


def main():
    print("""
╔═══════════════════════════════════════════════════════════════════════════╗
║              SAMAPy HYBRID CONFIGURATION WIZARD                             ║
╚═══════════════════════════════════════════════════════════════════════════╝
""")

    # Get current directory
    current_dir = Path.cwd()

    # Add samapy/cli to Python path so we can import
    cli_dir = current_dir / 'samapy' / 'cli'

    if not cli_dir.exists():
        print(f"❌ Error: {cli_dir} not found!")
        input("\nPress Enter to exit...")
        sys.exit(1)

    # Add to path
    sys.path.insert(0, str(cli_dir))

    # Import and run the wizard directly
    try:
        import samapy_Wizard

        print("\n🚀 Starting configuration wizard...\n")
        print("=" * 80 + "\n")

        # Run the wizard's main function
        samapy_Wizard.main()

    except KeyboardInterrupt:
        print("\n\n❌ Cancelled by user.")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        input("\nPress Enter to exit...")
        sys.exit(1)


if __name__ == '__main__':
    main()