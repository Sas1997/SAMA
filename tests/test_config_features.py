"""Demand charges and bi-facial modules set through the YAML configuration.

apply_config changes the global InData object, so each case runs in a
separate Python process.
"""
import json
import re
import subprocess
import sys
from pathlib import Path

import pytest

CONFIG = Path(__file__).resolve().parents[1] / "samapy_config_COMPLETE.yaml"

SCRIPT = r"""
import io, contextlib, json, sys
import numpy as np
from samapy.cli.config_loader import load_config, apply_config
from samapy.cli import runner
with contextlib.redirect_stdout(io.StringIO()):
    d = apply_config(load_config(sys.argv[1]))
    runner._patch_fitness(d)
import samapy.core.Fitness as F
m = F.evaluate(np.array([12.0, 0.0, 2.0, 0.0, 8.0]))
print(json.dumps({
    "NPC": m["NPC"], "dc": m["demand_charge_year1"],
    "rates": list(map(float, d.Demand_charge_rate_monthly)),
    "module": d.PV_module_type, "n_sub": len(d.G_list),
    "rear": [None if r is None else float(np.sum(r)) for r in d.G_rear_list],
}))
"""


def run_config(tmp_path, changes):
    text = CONFIG.read_text(encoding="utf-8")
    for key, value in changes.items():
        text, n = re.subn(rf"(?m)^{key}:.*$", f"{key}: {value}", text)
        assert n == 1, key
    path = tmp_path / "config.yaml"
    path.write_text(text, encoding="utf-8")
    out = subprocess.run([sys.executable, "-c", SCRIPT, str(path)], cwd=tmp_path,
                         capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=900)
    assert out.returncode == 0, out.stderr[-3000:]
    return json.loads(out.stdout.strip().splitlines()[-1])


pytestmark = pytest.mark.skipif(not CONFIG.is_file(), reason="example configuration not found")


def test_example_config_has_features_off(tmp_path):
    r = run_config(tmp_path, {})
    assert r["dc"] == 0.0 and not any(r["rates"])
    assert r["module"] == "Mono-facial" and r["rear"] == [None]


def test_config_enables_demand_charges(tmp_path):
    base = run_config(tmp_path, {})
    r = run_config(tmp_path, {"demand_charge_enabled": 1, "demand_charge_type": 3})
    assert r["rates"] == [12.0, 12.0, 14.0, 14.0, 16.0, 18.0, 18.0, 18.0, 16.0, 14.0, 12.0, 12.0]
    assert r["dc"] > 0 and r["NPC"] > base["NPC"]


def test_config_enables_bifacial_subarrays(tmp_path):
    base = run_config(tmp_path, {})
    r = run_config(tmp_path, {
        "PV_module_type": "Bi-facial",
        "PV_subarrays": "[{tilt: 33, azimuth: 180, share: 0.7}, {tilt: 90, azimuth: 270, share: 0.3}]"})
    assert r["module"] == "Bi-facial" and r["n_sub"] == 2
    assert all(x > 0 for x in r["rear"])
    assert r["NPC"] != base["NPC"]
