"""Input files named in the configuration (load, thermal load, weather data).

A relative name is looked up in the input data folder chosen in the wizard,
the folder of the configuration file and the current folder, then in SAMA's
example data; a full path is used as given. A file that cannot be found
stops the run instead of silently falling back to the example data.

apply_config changes the shared InData object, so each case runs in a
separate Python process.
"""
import json
import re
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from samapy import get_content_path
from samapy.cli.config_loader import resolve_input_path

CONFIG = Path(__file__).resolve().parents[1] / "samapy_config_COMPLETE.yaml"
pytestmark = pytest.mark.skipif(not CONFIG.is_file(), reason="example configuration not found")

SCRIPT = r"""
import io, contextlib, json, sys
import numpy as np
from samapy.cli.config_loader import load_config, apply_config
with contextlib.redirect_stdout(io.StringIO()):
    d = apply_config(load_config(sys.argv[1]))
out = {k: float(np.sum(np.asarray(getattr(d, k), dtype=float)))
       for k in ('Eload_eh', 'Eload_Previous', 'Hload', 'G', 'T', 'Vw', 'P')}
out['weather_url'] = str(d.weather_url)
print(json.dumps(out))
"""


def content_sum(name, column=0):
    return float(pd.read_csv(get_content_path(name), header=None).values[:, column].sum())


def write_config(folder, changes):
    """Example configuration with the given keys replaced or added."""
    text = CONFIG.read_text(encoding="utf-8")
    for key, value in changes.items():
        line = f"{key}: {json.dumps(value)}"
        text, n = re.subn(rf"(?m)^{key}:.*$", line.replace("\\", "\\\\"), text)
        if n == 0:
            text += f"\n{line}\n"
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / "samapy_config_COMPLETE.yaml"
    path.write_text(text, encoding="utf-8")
    return path


def run(config_path, cwd=None):
    r = subprocess.run([sys.executable, "-c", SCRIPT, str(config_path)], cwd=cwd or config_path.parent,
                       capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=900)
    return r


def values(config_path, cwd=None):
    r = run(config_path, cwd)
    assert r.returncode == 0, r.stderr[-3000:]
    return json.loads(r.stdout.strip().splitlines()[-1])


def scaled_copy(name, target, factor=2.0):
    target.parent.mkdir(parents=True, exist_ok=True)
    data = pd.read_csv(get_content_path(name), header=None).values[:, 0] * factor
    np.savetxt(target, data, delimiter=",")
    return target


def test_resolution_order(tmp_path):
    first, second = tmp_path / "a", tmp_path / "b"
    for folder in (first, second):
        folder.mkdir()
        (folder / "load.csv").write_text("1\n")
    (second / "only_b.csv").write_text("1\n")
    assert resolve_input_path("load.csv", [first, second]) == str((first / "load.csv").resolve())
    assert resolve_input_path("only_b.csv", [first, second]) == str((second / "only_b.csv").resolve())
    assert resolve_input_path(str(second / "load.csv"), [first]) == str(second / "load.csv")
    # standard names fall back to SAMA's example data, other names are not found
    assert resolve_input_path("Eload.csv", [first]) == get_content_path("Eload.csv")
    assert resolve_input_path("missing.csv", [first, second]) is None
    assert resolve_input_path(str(tmp_path / "missing.csv")) is None


@pytest.mark.parametrize("where", ["config_folder", "subfolder", "input_data_directory", "full_path"])
def test_custom_load_file_is_used(tmp_path, where):
    project, data = tmp_path / "project", tmp_path / "my data"
    if where == "config_folder":
        scaled_copy("Eload.csv", project / "myload.csv")
        name, extra = "myload.csv", {}
    elif where == "subfolder":
        scaled_copy("Eload.csv", project / "inputs" / "myload.csv")
        name, extra = "inputs/myload.csv", {}
    elif where == "input_data_directory":
        scaled_copy("Eload.csv", data / "myload.csv")
        name, extra = "myload.csv", {"input_data_directory": str(data)}
    else:
        name, extra = str(scaled_copy("Eload.csv", data / "myload.csv")), {}
    cfg = write_config(project, {"load_type": 1, "path_Eload": name, **extra})
    # run from another folder: the configuration's folder is still searched
    v = values(cfg, cwd=tmp_path)
    assert v["Eload_eh"] == pytest.approx(2 * content_sum("Eload.csv"))


def test_missing_file_stops_instead_of_using_example_data(tmp_path):
    cfg = write_config(tmp_path / "project", {"load_type": 1, "path_Eload": "not_there.csv"})
    r = run(cfg)
    assert r.returncode != 0
    assert "Input file(s) not found" in r.stderr and "not_there.csv" in r.stderr
    # samapy-run reports it and stops
    r = subprocess.run([sys.executable, "-m", "samapy.cli.runner", "-c", str(cfg), "--dry-run", "--no-gui"],
                       cwd=cfg.parent, capture_output=True, text=True, encoding="utf-8", errors="replace",
                       timeout=900)
    assert r.returncode == 1 and "Input file(s) not found" in r.stdout


def test_standard_names_use_the_example_data(tmp_path):
    v = values(write_config(tmp_path / "project", {"load_type": 1, "path_Eload": "Eload.csv"}))
    assert v["Eload_eh"] == pytest.approx(content_sum("Eload.csv"))
    assert Path(v["weather_url"]) == Path(get_content_path("METEO.csv"))


def test_every_csv_input_type_reads_the_custom_file(tmp_path):
    project = tmp_path / "project"
    for name in ("Irradiance.csv", "Temperature.csv", "WSPEED.csv"):
        scaled_copy(name, project / "inputs" / f"site_{name}")
    # SAMA has no example Eload_previousyear.csv: use its example load as the previous year
    scaled_copy("Eload.csv", project / "inputs" / "site_prev.csv")
    house = pd.read_excel(get_content_path("house_load.xlsx"))
    house.iloc[:, 1] *= 2
    house.to_excel(project / "inputs" / "site_house.xlsx", index=False)
    cfg = write_config(project, {
        "load_previous_year_type": 2, "path_Eload_Previous": "inputs/site_prev.csv",
        "G_type": 2, "path_G": "inputs/site_Irradiance.csv",
        "T_type": 2, "path_T": "inputs/site_Temperature.csv",
        "WS_type": 2, "path_WS": "inputs/site_WSPEED.csv",
        "path_house_load": "inputs/site_house.xlsx"})
    v = values(cfg)
    assert v["Eload_Previous"] == pytest.approx(2 * content_sum("Eload.csv"))
    assert v["G"] == pytest.approx(2 * content_sum("Irradiance.csv"))
    assert v["T"] == pytest.approx(2 * content_sum("Temperature.csv"))
    assert v["Vw"] == pytest.approx(2 * content_sum("WSPEED.csv"))
    default_heating = float(pd.read_excel(get_content_path("house_load.xlsx")).iloc[:, 1].sum())
    assert v["Hload"] == pytest.approx(2 * default_heating)


def test_daily_profile_file_with_a_custom_name(tmp_path):
    project = tmp_path / "project"
    (project / "inputs").mkdir(parents=True)
    (project / "inputs" / "my_daily.csv").write_bytes(Path(get_content_path("Eload_daily.csv")).read_bytes())
    custom = values(write_config(project, {"load_type": 10, "path_Eload_daily": "inputs/my_daily.csv"}))
    standard = values(write_config(tmp_path / "standard", {"load_type": 10, "path_Eload_daily": "Eload_daily.csv"}))
    assert custom["Eload_eh"] == pytest.approx(standard["Eload_eh"])


def test_weather_file_sets_irradiance_temperature_and_pressure(tmp_path):
    project = tmp_path / "project"
    project.mkdir()
    src = Path(get_content_path("METEO.csv"))
    head = src.read_text(encoding="utf-8").splitlines()[:2]
    data = pd.read_csv(src, header=2)
    data["Pressure"] = data["Pressure"] - 50
    data["Temperature"] = data["Temperature"] + 3
    target = project / "site weather.csv"
    with open(target, "w", encoding="utf-8", newline="") as f:
        f.write("\n".join(head) + "\n")
        data.to_csv(f, index=False)
    default = values(write_config(tmp_path / "default", {}))
    v = values(write_config(project, {"weather_url": "site weather.csv"}))
    assert Path(v["weather_url"]) == target.resolve()
    assert v["P"] == pytest.approx(default["P"] - 50 * 8760)
    assert v["T"] == pytest.approx(default["T"] + 3 * 8760, rel=1e-6)
