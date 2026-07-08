"""
SAMAPy CLI - Wizard Entry Point
==============================
Entry point for the `samapy-config` console script defined in setup.py.

Launches the interactive samapy_Wizard configuration wizard which guides
the user through all system parameters and produces a YAML config file
ready for use with `samapy-run`.

Usage (after pip install samapy):
    samapy-config

Usage (direct):
    python -m samapy.cli.wizard
"""

import sys
from pathlib import Path


def _check_questionary():
    """Check questionary is installed and give a clear error if not."""
    try:
        import questionary  # noqa: F401
    except ImportError:
        print("❌ Missing dependency: questionary")
        print()
        print("   Install it with:")
        print("       pip install questionary")
        print()
        sys.exit(1)


def main():
    print("""
╔═══════════════════════════════════════════════════════════════════════════╗
║                 SAMAPy - Hybrid Energy System Optimization                ║
║                   samapy-config  |  Configuration Wizard                  ║
╚═══════════════════════════════════════════════════════════════════════════╝
""")

    # Verify questionary is available before attempting to import the wizard
    _check_questionary()

    # samapy_Wizard.py lives alongside this file in samapy/cli/
    cli_dir = Path(__file__).parent

    # Add cli directory to path so samapy_Wizard can be imported directly
    if str(cli_dir) not in sys.path:
        sys.path.insert(0, str(cli_dir))

    try:
        import samapy_Wizard
        print("🚀 Starting configuration wizard...\n")
        print("=" * 78 + "\n")
        samapy_Wizard.main()

    except ImportError:
        # Fallback: try importing as a module inside the package
        try:
            from samapy.cli import samapy_Wizard  # type: ignore
            samapy_Wizard.main()
        except ImportError:
            print("❌ Could not find samapy_Wizard.py")
            print(f"   Expected location: {cli_dir / 'samapy_Wizard.py'}")
            print()
            print("Please ensure the SAMAPy package is correctly installed:")
            print("    pip install samapy")
            sys.exit(1)

    except KeyboardInterrupt:
        print("\n\n⚠️  Configuration wizard cancelled by user.")
        sys.exit(1)

    except Exception as e:
        print(f"\n❌ Wizard error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()
