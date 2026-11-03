"""The reference case of the documentation (Section 18) reproduces its published results.

examples/reference_config.yaml is run with samapy-run and seed 1, and the
system size and key results are compared with the values printed in
Docs/SAMA_Documentation.md, Section 18.
"""
import re
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
REFERENCE = ROOT / "examples" / "reference_config.yaml"

EXPECTED = {
    r"Cpv  \(kW\) = ([\d.]+)": 12.0,
    r"Cbat \(kWh\) = ([\d.]+)": 0.0,
    r"Cinverter \(kW\) = ([\d.]+)": 10.66,
    r"NPC  = \$ ([\d.]+)": 67259.19,
    r"Total Money paid by the user= \$ ([\d.]+)": 67259.19,
    r"Total Money paid to the Grid= \$ ([\d.]+)": 31779.64,
    r"Initial Cost  = \$ ([\d.]+)": 7533.58,
    r"RE  = ([\d.]+) %": 61.32,
}


@pytest.mark.slow
@pytest.mark.skipif(not REFERENCE.is_file(), reason="reference configuration not found")
def test_reference_case_matches_documentation(tmp_path):
    r = subprocess.run([sys.executable, "-c", "from samapy.cli.runner import main; main()",
                        "-c", str(REFERENCE), "--seed", "1", "--no-gui", "--output", str(tmp_path / "out")],
                       cwd=tmp_path, capture_output=True, text=True, encoding="utf-8", errors="replace",
                       timeout=3600)
    assert r.returncode == 0, r.stdout[-3000:] + r.stderr[-3000:]
    for pattern, value in EXPECTED.items():
        m = re.search(pattern, r.stdout)
        assert m, f"not found in the output: {pattern}"
        assert float(m.group(1)) == pytest.approx(value, abs=0.01), pattern
    assert "Iteration 200, Best Cost = 0.6726" in r.stdout
    assert (tmp_path / "out" / "data" / "Outputforplotting.csv").is_file()


def test_fixture_configuration_is_the_reference_case():
    """The tests use their own copy of the reference configuration (tests/data), so running the
    wizard in the repository root, which rewrites samapy_config_COMPLETE.yaml there, does not
    change what the tests check."""
    fixture = Path(__file__).resolve().parent / "data" / "example_config.yaml"
    if not REFERENCE.is_file():
        pytest.skip("reference configuration not found")
    assert REFERENCE.read_text(encoding="utf-8") == fixture.read_text(encoding="utf-8")
