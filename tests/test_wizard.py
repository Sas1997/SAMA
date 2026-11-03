"""The configuration wizard (samapy-config) with scripted answers.

Checks that the optional features (multi-objective mode, bi-facial PV and PV
sub-arrays, demand charges, minimum charge) are off when every question is
answered with its default, and that answers turning them on reach the backend
exactly: the YAML written by the wizard is applied with apply_config in a
separate Python process (it changes the shared InData object) and the
resulting InData values are compared with the answers.
"""
import json
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

questionary = pytest.importorskip("questionary")
from samapy.cli import samapy_Wizard as W  # noqa: E402


class _Answer:
    def __init__(self, value):
        self.value = value

    def ask(self):
        return self.value


class Seq:
    """Answers given in turn when a question is asked more than once."""
    def __init__(self, *answers):
        self.answers = list(answers)

    def next(self):
        assert self.answers, "question asked more often than expected"
        return self.answers.pop(0)


def run_wizard(tmp_path, monkeypatch, answers):
    """Run the wizard in tmp_path; unanswered questions take their default."""
    used = set()

    def lookup(message):
        for key, value in answers.items():
            if key in message:
                used.add(key)
                return True, (value.next() if isinstance(value, Seq) else value)
        return False, None

    def text(message, default="", validate=None, **kw):
        found, value = lookup(message)
        value = str(value) if found else default
        if validate is not None:
            assert validate(value) is True, (message, value, validate(value))
        return _Answer(value)

    def select(message, choices, **kw):
        found, value = lookup(message)
        values = [c.value for c in choices]
        if found:
            assert value in values, (message, value, values)
            return _Answer(value)
        return _Answer(values[0])

    def confirm(message, default=False, **kw):
        found, value = lookup(message)
        return _Answer(value if found else default)

    def checkbox(message, choices, validate=None, **kw):
        found, value = lookup(message)
        value = value if found else [c.value for c in choices if c.checked]
        if validate is not None:
            assert validate(value) is True, (message, value)
        return _Answer(value)

    for name, fn in (("text", text), ("select", select), ("confirm", confirm), ("checkbox", checkbox)):
        monkeypatch.setattr(W.questionary, name, fn)
    monkeypatch.chdir(tmp_path)
    W.main()
    assert used == set(answers), f"questions never asked: {set(answers) - used}"
    path = tmp_path / "samapy_config_COMPLETE.yaml"
    return path, yaml.safe_load(path.read_text())


APPLY = r"""
import io, contextlib, json, sys
import numpy as np
from samapy.cli.config_loader import load_config, apply_config
from samapy.cli import runner
with contextlib.redirect_stdout(io.StringIO()):
    d = apply_config(load_config(sys.argv[1]))
    runner._patch_fitness(d)
import samapy.core.Fitness as F
m = F.evaluate(np.array([float(v) for v in sys.argv[2].split(',')]))
keys = ['MOO_objectives', 'MOO_selection', 'MOO_n_pop', 'MOO_n_gen', 'PV_module_type', 'bifaciality',
        'shade_factor', 'pv_gcr', 'pv_height', 'pv_pitch', 'PV_subarrays', 'PV_shares',
        'demand_charge_enabled', 'demand_charge_type', 'demand_tou_weekdays_only', 'demand_ratchet_enabled',
        'demand_ratchet_rate', 'nem_demand_netting', 'nem_demand_rate_override', 'minimum_charge_enabled',
        'minimum_charge_credit_carryover', 'Demand_charge_rate_monthly', 'Demand_charge_offpeak_rate_monthly',
        'demand_tou_season', 'minimum_charge_monthly', 'demand_tou_onpeak_hours']
out = {}
for k in keys:
    v = getattr(d, k)
    if isinstance(v, np.ndarray):
        v = [list(map(int, x)) if isinstance(x, (list, np.ndarray)) else float(x) for x in v]
    out[k] = v
out['rear'] = [None if r is None else float(np.sum(r)) for r in d.G_rear_list]
out['Eload_eh'] = float(np.sum(d.Eload_eh))
out['P'] = float(np.sum(d.P))
out['NPC'] = m['NPC']
out['dc'] = m['demand_charge_year1']
print(json.dumps(out))
"""


def apply(path, design="12,0,2,0,8"):
    r = subprocess.run([sys.executable, "-c", APPLY, str(path), design], cwd=path.parent,
                       capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=900)
    assert r.returncode == 0, r.stderr[-3000:]
    return json.loads(r.stdout.strip().splitlines()[-1])


def test_defaults_keep_every_new_feature_off(tmp_path, monkeypatch):
    path, cfg = run_wizard(tmp_path, monkeypatch, {})
    assert cfg["optimization_mode"] == "single"
    assert cfg["MOO_objectives"] == ["npc", "re"] and cfg["MOO_selection"] == "compromise"
    assert "MOO_n_pop" not in cfg and "MOO_n_gen" not in cfg
    assert cfg["PV_module_type"] == "Mono-facial" and cfg["PV_subarrays"] is None
    assert cfg["demand_charge_enabled"] == 0 and cfg["minimum_charge_enabled"] == 0
    assert cfg["nem_demand_netting"] == 0 and cfg["nem_demand_rate_override"] == 0

    # The backend result is identical to a configuration without the new keys
    new = apply(path)
    old_cfg = {k: v for k, v in cfg.items() if k not in W.DEMAND_DEFAULTS and k not in W.PV_GEOMETRY_DEFAULTS
               and k not in ('optimization_mode', 'MOO_objectives', 'MOO_selection',
                             'PV_module_type', 'PV_subarrays')}
    old_path = tmp_path / "old.yaml"
    old_path.write_text(yaml.dump(old_cfg, sort_keys=False))
    old = apply(old_path)
    assert new["NPC"] == old["NPC"]
    assert new["dc"] == 0.0 and new["rear"] == [None] and new["PV_shares"] == [1.0]
    assert not any(new["Demand_charge_rate_monthly"]) and new["minimum_charge_enabled"] == 0


FEATURES_ON = {
    "Optimization mode": "multi",
    "Objectives (space": ["npc", "lcoe", "re"],
    "Design from the Pareto front": "min_npc",
    "NSGA-II population size": 20,
    "NSGA-II number of generations": 5,
    "PV module type": "Bi-facial",
    "Bifaciality factor": 0.8,
    "Rear shading factor": -0.05,
    "Ground coverage ratio": 0.4,
    "Height of the row centre": 2.0,
    "Row pitch": 5.0,
    "Split the PV capacity": True,
    "Number of sub-arrays": 2,
    "Sub-array 1: tilt": 90, "Sub-array 1: azimuth": 90, "Sub-array 1: share": 0.6,
    "Sub-array 2: tilt": 90, "Sub-array 2: azimuth": 270, "Sub-array 2: share": 0.4,
    "Does the tariff include demand charges": True,
    "Demand charge type": 4,
    "On-peak demand rate - Summer": 22, "On-peak demand rate - Winter": 16,
    "Off-peak demand rate - Summer": 3, "Off-peak demand rate - Winter": 2,
    "12 season indicators": "0,0,0,0,1,1,1,1,1,0,0,0",
    "On-peak demand hours - Summer": "14,15,16,17,18",
    "On-peak demand hours - Winter": "7,8,17,18",
    "Apply the on-peak demand hours on weekdays only": True,
    "Apply a demand ratchet": True,
    "Ratchet fraction": 0.9,
    "Measure the demand peak on the net import": True,
    "separate demand rates to net-metered": True,
    "12 monthly NEM on-peak demand rates": "11,11,11,11,12,12,12,12,12,11,11,11",
    "12 monthly NEM off-peak demand rates": "1,1,1,1,2,2,2,2,2,1,1,1",
    "minimum monthly charge": True,
    "12 monthly minimum charges": "30,30,30,30,30,30,30,30,30,30,30,30",
    "Carry export credit": True,
}


def test_answers_turn_the_features_on_in_the_backend(tmp_path, monkeypatch):
    path, cfg = run_wizard(tmp_path, monkeypatch, FEATURES_ON)
    assert cfg["optimization_mode"] == "multi"
    r = apply(path)
    assert r["MOO_objectives"] == ["npc", "lcoe", "re"] and r["MOO_selection"] == "min_npc"
    assert (r["MOO_n_pop"], r["MOO_n_gen"]) == (20, 5)
    assert r["PV_module_type"] == "Bi-facial"
    assert (r["bifaciality"], r["shade_factor"], r["pv_gcr"], r["pv_height"], r["pv_pitch"]) == (0.8, -0.05, 0.4, 2.0, 5.0)
    assert r["PV_subarrays"] == [{"tilt": 90.0, "azimuth": 90.0, "share": 0.6},
                                 {"tilt": 90.0, "azimuth": 270.0, "share": 0.4}]
    assert r["PV_shares"] == [0.6, 0.4] and all(x > 0 for x in r["rear"])
    assert (r["demand_charge_enabled"], r["demand_charge_type"], r["demand_tou_weekdays_only"]) == (1, 4, 1)
    assert (r["demand_ratchet_enabled"], r["demand_ratchet_rate"]) == (1, 0.9)
    assert (r["nem_demand_netting"], r["nem_demand_rate_override"]) == (1, 1)
    assert r["demand_tou_season"] == [0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0]
    assert r["demand_tou_onpeak_hours"] == [[14, 15, 16, 17, 18], [7, 8, 17, 18]]
    # Option C: the NEM rates replace the TOU rates entered above
    assert r["Demand_charge_rate_monthly"] == [11, 11, 11, 11, 12, 12, 12, 12, 12, 11, 11, 11]
    assert r["Demand_charge_offpeak_rate_monthly"] == [1, 1, 1, 1, 2, 2, 2, 2, 2, 1, 1, 1]
    assert (r["minimum_charge_enabled"], r["minimum_charge_credit_carryover"]) == (1, 1)
    assert r["minimum_charge_monthly"] == [30.0] * 12
    assert r["dc"] > 0


@pytest.mark.parametrize("answers, rates", [
    ({"Demand charge type": 1, "Demand rate ($/kW/month)": 9.5}, [9.5] * 12),
    ({"Demand charge type": 2, "Demand rate - Summer": 20, "Demand rate - Winter": 10,
      "12 season indicators": "0,0,0,0,0,1,1,1,0,0,0,0"}, [10] * 5 + [20] * 3 + [10] * 4),
    ({"Demand charge type": 3, "12 monthly demand rates": "1,2,3,4,5,6,7,8,9,10,11,12"}, list(range(1, 13))),
    ({"Demand charge type": 4, "On-peak demand rate - Summer": 30, "On-peak demand rate - Winter": 25,
      "Off-peak demand rate - Summer": 5, "Off-peak demand rate - Winter": 4,
      "12 season indicators": "0,0,0,0,0,1,1,1,1,0,0,0"}, [25] * 5 + [30] * 4 + [25] * 3),
])
def test_demand_charge_types_reach_the_monthly_rates(tmp_path, monkeypatch, answers, rates):
    answers = {"Does the tariff include demand charges": True, **answers}
    path, cfg = run_wizard(tmp_path, monkeypatch, answers)
    r = apply(path)
    assert r["Demand_charge_rate_monthly"] == [float(x) for x in rates]
    assert r["dc"] > 0
    if answers["Demand charge type"] == 4:
        assert r["Demand_charge_offpeak_rate_monthly"] == [4.0] * 5 + [5.0] * 4 + [4.0] * 3
        assert r["demand_tou_weekdays_only"] == 0
    else:
        assert not any(r["Demand_charge_offpeak_rate_monthly"])


def test_offgrid_asks_no_net_metering_or_minimum_charge(tmp_path, monkeypatch):
    path, cfg = run_wizard(tmp_path, monkeypatch, {"Include Grid Connection?": False,
                                                   "Does the tariff include demand charges": True})
    # Demand charges can still be set for the grid-only comparison, NEM options and
    # the minimum charge (which needs a grid connection) are not asked
    assert cfg["demand_charge_enabled"] == 1
    assert cfg["nem_demand_netting"] == 0 and cfg["nem_demand_rate_override"] == 0
    assert cfg["minimum_charge_enabled"] == 0


def test_bifacial_needs_the_weather_file(tmp_path, monkeypatch):
    path, cfg = run_wizard(tmp_path, monkeypatch, {"Select irradiance input method": 2})
    assert cfg["G_type"] == 2
    assert cfg["PV_module_type"] == "Mono-facial" and cfg["PV_subarrays"] is None


def test_weights_selection_and_validation(tmp_path, monkeypatch):
    path, cfg = run_wizard(tmp_path, monkeypatch, {
        "Optimization mode": "multi", "Design from the Pareto front": "weights",
        "Weights for npc, re": "2,1"})
    assert cfg["MOO_selection"] == [2.0, 1.0]
    assert cfg["MOO_n_pop"] == cfg["nPop"] and cfg["MOO_n_gen"] == cfg["MaxIt"]
    check = W._check_list(n=12, kind=int, lo=0, hi=1)
    assert check("0,1,0,1,0,1,0,1,0,1,0,1") is True
    assert check("0,1,2,1,0,1,0,1,0,1,0,1") is not True
    assert check("0,1") is not True
    assert W._check_number(lo=0, lo_open=True)("0") is not True


def test_runner_rejects_invalid_multiobjective_settings(tmp_path, monkeypatch):
    path, cfg = run_wizard(tmp_path, monkeypatch, {"Optimization mode": "multi"})
    cfg["MOO_selection"] = "min_lem"          # not one of the chosen objectives
    path.write_text(yaml.dump(cfg, sort_keys=False))
    r = subprocess.run([sys.executable, "-m", "samapy.cli.runner", "-c", str(path), "--dry-run", "--no-gui"],
                       cwd=tmp_path, capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=900)
    assert r.returncode == 1 and "Multi-objective settings" in r.stdout


def _content_hashes():
    import hashlib
    from samapy import get_content_path
    root = Path(get_content_path(""))
    return {f: hashlib.sha256(f.read_bytes()).hexdigest() for f in root.rglob("*") if f.is_file()}


def test_input_files_from_the_chosen_folder(tmp_path, monkeypatch):
    import numpy as np
    import pandas as pd
    from samapy import get_content_path
    data = tmp_path / "my inputs"
    data.mkdir()
    load = pd.read_csv(get_content_path("Eload.csv"), header=None).values[:, 0]
    np.savetxt(data / "myload.csv", 2 * load, delimiter=",")
    np.savetxt(data / "prev.csv", load, delimiter=",")
    before = _content_hashes()
    project = tmp_path / "project"
    project.mkdir()
    path, cfg = run_wizard(project, monkeypatch, {
        "Folder with your input files": str(data),
        "Select electrical load input type": 1,
        # a wrong name is asked again; then the file in the chosen folder is used
        "Hourly electrical load file": Seq("nope.csv", "myload.csv"),
        "Select previous year load type": 2,
        # SAMA has no example previous-year load, so the standard name is asked again too
        "Previous-year hourly load file": Seq("Eload_previousyear.csv", "prev.csv"),
    })
    assert cfg["input_data_directory"] == str(data.resolve())
    assert cfg["path_Eload"] == str((data / "myload.csv").resolve())
    assert cfg["path_Eload_Previous"] == str((data / "prev.csv").resolve())
    # files kept at their standard names use SAMA's example data
    assert cfg["weather_url"] == "METEO.csv" and cfg["path_house_load"] == "house_load.xlsx"
    assert _content_hashes() == before, "the wizard must not change SAMA's own files"
    r = apply(path)
    assert r["Eload_eh"] == pytest.approx(2 * load.sum())


def test_weather_file_asked_when_temperature_or_heat_pump_needs_it(tmp_path, monkeypatch):
    (tmp_path / "a").mkdir()
    path, cfg = run_wizard(tmp_path / "a", monkeypatch, {
        "Select irradiance input method": 2,
        "Weather file for temperature / wind speed": "METEO.csv"})
    assert cfg["G_type"] == 2 and cfg["weather_url"] == "METEO.csv"
    (tmp_path / "b").mkdir()
    path, cfg = run_wizard(tmp_path / "b", monkeypatch, {
        "Select irradiance input method": 2,
        "Select temperature input method": 4,
        "Select wind speed input method": 4,
        "Include Heat Pump?": True,
        "Weather file for the air pressure": "METEO.csv"})
    assert (cfg["T_type"], cfg["WS_type"], cfg["HP"]) == (4, 4, 1) and cfg["weather_url"] == "METEO.csv"
