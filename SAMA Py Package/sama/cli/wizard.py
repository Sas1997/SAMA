"""
SAMA CLI - Wizard Entry Point
==============================
Entry point for the `sama-config` console script defined in setup.py.

Launches the interactive sama_Wizard configuration wizard which guides
the user through all system parameters and produces a YAML config file
ready for use with `sama-run`.

Usage (after pip install sama):
    sama-config

Usage (direct):
    python -m sama.cli.wizard
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
║              SAMA - Hybrid Energy System Optimization                     ║
║                   sama-config  |  Configuration Wizard                    ║
╚═══════════════════════════════════════════════════════════════════════════╝
""")

    # Verify questionary is available before attempting to import the wizard
    _check_questionary()

    # sama_Wizard.py lives alongside this file in sama/cli/
    cli_dir = Path(__file__).parent

    # Add cli directory to path so sama_Wizard can be imported directly
    if str(cli_dir) not in sys.path:
        sys.path.insert(0, str(cli_dir))

    try:
        import sama_Wizard
        print("🚀 Starting configuration wizard...\n")
        print("=" * 78 + "\n")
        sama_Wizard.main()

    except ImportError:
        # Fallback: try importing as a module inside the package
        try:
            from sama.cli import sama_Wizard  # type: ignore
            sama_Wizard.main()
        except ImportError:
            print("❌ Could not find sama_Wizard.py")
            print(f"   Expected location: {cli_dir / 'sama_Wizard.py'}")
            print()
            print("Please ensure the SAMA package is correctly installed:")
            print("    pip install sama")
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
