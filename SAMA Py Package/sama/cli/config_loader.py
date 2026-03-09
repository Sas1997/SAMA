"""
SAMA CLI ─ config_loader.py
======================================================

"""

import yaml
import numpy as np
from pathlib import Path
from math import ceil


# ─────────────────────────────────────────────────────────────────────────────
# Public helpers
# ─────────────────────────────────────────────────────────────────────────────

def load_config(path):
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Config file not found: {p}")
    with open(p, "r") as f:
        cfg = yaml.safe_load(f)
    return cfg if cfg else {}


def save_config(config, path):
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w") as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)
    print(f"✅ Config saved → {p}")


def merge_configs(*paths):
    merged = {}
    for path in paths:
        try:
            merged.update(load_config(path))
        except FileNotFoundError:
            pass
    return merged


# ─────────────────────────────────────────────────────────────────────────────
# Type helpers
# ─────────────────────────────────────────────────────────────────────────────

def _to_np_array(val, dtype=float):
    if isinstance(val, np.ndarray):
        return val.astype(dtype)
    if isinstance(val, list):
        return np.array(val, dtype=dtype)
    return np.array([val], dtype=dtype)


def _to_hours_object_array(val):
    """
    Convert YAML hours (list-of-lists, possibly unequal lengths) to a numpy
    object array where every element is a plain Python list.
    Used for storing onHours/midHours/ultraLowHours on InData (d.*).
    """
    if isinstance(val, np.ndarray) and val.dtype == object:
        out = np.empty(len(val), dtype=object)
        for i, row in enumerate(val):
            out[i] = list(row) if not isinstance(row, list) else row
        return out
    if isinstance(val, list):
        out = np.empty(len(val), dtype=object)
        for i, row in enumerate(val):
            if isinstance(row, list):
                out[i] = row
            elif isinstance(row, np.ndarray):
                out[i] = row.tolist()
            else:
                out[i] = [row]
        return out
    return val


def _to_equal_hours_array(val):
    """
    Convert list-of-lists hours to a proper 2D numpy int array that
    calcTouRate.py can index with both onHours[0] AND onHours[1, :].

    WHY EQUAL-LENGTH ROWS ARE REQUIRED:
    calcTouRate uses two different indexing styles:
        Summer: onHours[0]      (object-array style — works with any format)
        Winter: onHours[1, :]   (2D numpy style — requires rectangular array)
    When rows have unequal length, np.array([[...],[...]]) produces a 1-D
    object array, and onHours[1, :] raises "too many indices" error.

    HOW WE PAD SAFELY:
    calcTouRate first sets ALL hours in the month to off-peak, then overwrites
    only the on-peak/mid-peak hours. Padding shorter rows by repeating their
    last valid hour means those hours get written twice with the same price —
    a harmless no-op. We never pad with -1 or invalid values since
    calcTouRate uses each hour value as an array index: t_index[hour] + 24*d.
    """
    # Normalise to list-of-lists of ints
    if isinstance(val, np.ndarray):
        if val.dtype == object:
            rows = [[int(h) for h in r] if not isinstance(r, list)
                    else [int(h) for h in r] for r in val]
        elif val.ndim == 2:
            return val.astype(int)  # already rectangular
        else:
            rows = [[int(h) for h in val]]
    elif isinstance(val, list):
        rows = []
        for r in val:
            if isinstance(r, (list, np.ndarray)):
                rows.append([int(h) for h in r])
            else:
                rows.append([int(r)])
    else:
        return np.array([[int(val)]], dtype=int)

    if not rows:
        return np.empty((0, 0), dtype=int)

    max_len = max(len(r) for r in rows)
    out = np.empty((len(rows), max_len), dtype=int)
    for i, r in enumerate(rows):
        if len(r) == 0:
            # Empty hour list — fill with a dummy that repeats if needed
            # (calcTouRate checks len > 0 before iterating, so this row
            #  won't be iterated; we just need a valid shape)
            out[i, :] = 0
        else:
            out[i, :len(r)] = r
            # Pad by repeating the last valid hour — harmless double-write
            out[i, len(r):] = r[-1]
    return out


# ─────────────────────────────────────────────────────────────────────────────
# NG unit conversion  $/m³ → $/kWh
# ─────────────────────────────────────────────────────────────────────────────

def _cm2kwh(value, ng_ec, f_eff):
    if isinstance(value, (list, np.ndarray)):
        return np.array(value, dtype=float) / (ng_ec * f_eff)
    return float(value) / (ng_ec * f_eff)


# ─────────────────────────────────────────────────────────────────────────────
# Derived-parameter calculators (called in strict dependency order)
# ─────────────────────────────────────────────────────────────────────────────

def _recalc_days_in_month(d):
    try:
        from sama.utilities.daysInMonth import daysInMonth
        d.daysInMonth = daysInMonth(d.year)
    except Exception as e:
        print(f"  ⚠️  daysInMonth failed: {e}")


def _recalc_financial(d):
    d.ir            = (d.n_ir - d.e_ir) / (1 + d.e_ir)
    d.System_Tax    = d.Tax_rate / 100
    d.RE_incentives = d.RE_incentives_rate / 100


def _recalc_constraints(d):
    # LPSP_max_rate is stored as a percentage (e.g. 0.0999 means 0.0999%)
    # Input_Data computes LPSP_max = LPSP_max_rate / 100
    d.LPSP_max = d.LPSP_max_rate / 100
    d.RE_min   = d.RE_min_rate   / 100


def _recalc_grid_tax(d):
    if hasattr(d, 'Grid_sale_tax_rate'):
        d.Grid_Tax    = d.Grid_sale_tax_rate    / 100
    if hasattr(d, 'Grid_sale_tax_rate_NG'):
        d.Grid_Tax_NG = d.Grid_sale_tax_rate_NG / 100


def _recalc_grid_escalation(d):
    if hasattr(d, 'Grid_escalation_rate'):
        d.Grid_escalation    = np.array(d.Grid_escalation_rate,    dtype=float) / 100
    if hasattr(d, 'Grid_escalation_rate_NG'):
        d.Grid_escalation_NG = np.array(d.Grid_escalation_rate_NG, dtype=float) / 100


def _recalc_replacement_times(d):
    for life, rt in [('L_PV','RT_PV'), ('L_I','RT_I'), ('L_WT','RT_WT'),
                     ('L_B','RT_B'),   ('L_CH','RT_CH'),('L_HP','RT_HP'),
                     ('L_EV','RT_EV')]:
        if hasattr(d, life):
            setattr(d, rt, max(0, ceil(d.n / getattr(d, life)) - 1))


def _recalc_battery_rated_capacity(d):
    if getattr(d, 'Lead_acid', 0) == 1 and hasattr(d, 'Vnom_leadacid'):
        d.Cbt_r = (d.Vnom_leadacid * d.Cnom_Leadacid) / 1000
    elif getattr(d, 'Li_ion', 0) == 1 and hasattr(d, 'Vnom_Li_ion'):
        d.Cbt_r = (d.Vnom_Li_ion * d.Cnom_Li) / 1000


def _recalc_ev_derived(d):
    if all(hasattr(d, a) for a in ('C_ev', 'SOCe_min', 'SOCe_max')):
        d.C_ev_usable = d.C_ev * (d.SOCe_max - d.SOCe_min)
    if all(hasattr(d, a) for a in ('SOC_dep','Daily_trip','C_ev_usable','Range_EV','C_ev')):
        d.SOC_arr = d.SOC_dep - (d.Daily_trip * d.C_ev_usable) / (d.Range_EV * d.C_ev)


def _recalc_ev_battery_throughput(d):
    try:
        from sama.utilities.Ev_Battery_Throughput import calculate_ev_battery_throughput
        _, d.Q_lifetime_ev = calculate_ev_battery_throughput(
            d.C_ev_usable, d.degradation_percent, d.L_EV_dis, d.Range_EV, d.step_km)
    except Exception:
        pass


def _recalc_ev_presence(d):
    try:
        from sama.utilities.EV_Presence import determine_EV_presence
        d.EV_p = determine_EV_presence(
            d.year, d.Tout, d.Tin, d.holidays,
            getattr(d, 'treat_special_days_as_home', False))
    except Exception:
        pass


def _recalc_service_charge(d):
    system = getattr(d, 'Monthly_fixed_charge_system', 1)
    if system == 1:
        d.Service_charge = np.ones(12) * getattr(d, 'SC_flat', 15.0)
    elif system == 2:
        try:
            from sama.pricing.service_charge import service_charge
            eload_prev = getattr(d, 'Eload_Previous',
                         getattr(d, 'EloadPrevious', d.Eload))
            d.Service_charge = service_charge(
                d.daysInMonth, eload_prev,
                d.Limit_SC_1, d.SC_1, d.Limit_SC_2, d.SC_2,
                d.Limit_SC_3, d.SC_3, d.SC_4)
        except Exception as e:
            print(f"  ⚠️  service_charge failed: {e}")



def _recalc_service_charge_ng(d):
    system = getattr(d, 'Monthly_fixed_charge_system_NG', 1)
    if system == 1:
        d.Service_charge_NG = np.ones(12) * getattr(d, 'SC_flat_NG', 18.59)



def _recalc_nem_invariant(d):
    """Enforce: NEM must be 0 when Grid is 0."""
    if getattr(d, 'Grid', 1) == 0:
        d.NEM = 0


def _recalc_pricing_method(d, config):
    method = getattr(d, 'Pricing_method', 2)
    if method == 1 and 'Total_PV_price' in config:
        try:
            from sama.utilities.top_down import top_down_pricing
            d.Engineering_Costs, d.C_PV, d.R_PV, d.C_I, d.R_I = \
                top_down_pricing(float(config['Total_PV_price']))
            if 'MO_PV' in config:
                d.MO_PV = float(config['MO_PV'])
        except Exception as e:
            print(f"  ⚠️  top_down_pricing failed: {e}")
    elif method == 2:
        # Bottom-up: recompute Engineering_Costs from sub-fields (safe fallback to 0)
        fields = ['Sales_tax','Profit_costs','Fieldwork','Officework','Other',
                  'Permiting_and_Inspection','Electrical_BoS','Structrual_BoS',
                  'Supply_Chain_costs']
        d.Engineering_Costs = sum(getattr(d, f, 0.0) for f in fields)


# ─────────────────────────────────────────────────────────────────────────────
# Weather rebuild
# ─────────────────────────────────────────────────────────────────────────────

def _recalc_weather(d, config):
    weather_url = config.get('weather_url')
    tilt    = config.get('tilt',    getattr(d, 'tilt',    33))
    azimuth = config.get('azimuth', getattr(d, 'azimuth', 180))
    soiling = config.get('soiling', getattr(d, 'soiling', 5))
    _cache  = {}

    def get_sam():
        if 'r' not in _cache:
            from sama.utilities.sam_monofacial_poa import runSimulation
            from sama import get_content_path
            try:
                url = get_content_path(weather_url)
            except Exception:
                url = weather_url
            _cache['r'] = runSimulation(url, tilt, azimuth, soiling)
        return _cache['r']

    g_type  = config.get('G_type')
    t_type  = config.get('T_type')
    ws_type = config.get('WS_type')

    if g_type == 1:
        try:
            d.G = get_sam()[0].values
        except Exception as e:
            print(f"  ⚠️  G from SAM: {e}")
    elif g_type == 2:
        try:
            import pandas as pd; from sama import get_content_path
            d.G = np.array(pd.read_csv(
                get_content_path(config.get('path_G','Irradiance.csv')),
                header=None).values[:, 0])
        except Exception as e:
            print(f"  ⚠️  G from CSV: {e}")

    if t_type == 1:
        try:
            d.T = get_sam()[1].values
        except Exception as e:
            print(f"  ⚠️  T from SAM: {e}")
    elif t_type == 2:
        try:
            import pandas as pd; from sama import get_content_path
            d.T = np.array(pd.read_csv(
                get_content_path(config.get('path_T','Temperature.csv')),
                header=None).values[:, 0])
        except Exception as e:
            print(f"  ⚠️  T from CSV: {e}")
    elif t_type == 3:
        try:
            from sama.utilities.dataextender import dataextender
            d.T = dataextender(d.daysInMonth,
                  _to_np_array(config['Monthly_average_temperature']))
        except Exception as e:
            print(f"  ⚠️  T monthly: {e}")
    elif t_type == 4:
        d.T = np.full(8760, float(config.get('Annual_average_temperature', 12)))

    if ws_type == 1:
        try:
            d.Vw = get_sam()[2].values
        except Exception as e:
            print(f"  ⚠️  Vw from SAM: {e}")
    elif ws_type == 2:
        try:
            import pandas as pd; from sama import get_content_path
            d.Vw = np.array(pd.read_csv(
                get_content_path(config.get('path_WS','WSPEED.csv')),
                header=None).values[:, 0])
        except Exception as e:
            print(f"  ⚠️  Vw from CSV: {e}")
    elif ws_type == 3:
        try:
            from sama.utilities.dataextender import dataextender
            d.Vw = dataextender(d.daysInMonth,
                   _to_np_array(config['Monthly_average_windspeed']))
        except Exception as e:
            print(f"  ⚠️  Vw monthly: {e}")
    elif ws_type == 4:
        d.Vw = np.full(8760, float(config.get('Annual_average_windspeed', 10)))


# ─────────────────────────────────────────────────────────────────────────────
# HP load rebuild (FIX-14)
# Must run after weather (needs d.T, d.P) and before Eload is finalised.
# ─────────────────────────────────────────────────────────────────────────────

def _recalc_hp_load(d, config):
    """
    Re-run the HP model if HP==1 and HP_brand is present in config.
    Captures ALL return values from Heat_Pump_Model — not just Eload_hp —
    so that hp_model, HP_size, COP arrays, and power arrays are all updated
    to match the chosen brand. Without this, __init__ runs Goodman by default
    and those values stay stale even after HP_brand is changed to Bosch.

    Heat_Pump_Model returns:
        Eload_hp, power_hp_heating, power_hp_cooling,
        COP_hp_heating, COP_hp_cooling, hp_model, HP_size
    """
    if getattr(d, 'HP', 0) != 1:
        d.Eload_hp = 0
        return
    brand = getattr(d, 'HP_brand', 'Bosch')
    try:
        if brand == 'Goodman':
            from sama.models.BB_HP_Goodman import Heat_Pump_Model
            (eload_hp, d.power_hp_heating, d.power_hp_cooling,
             d.COP_hp_heating, d.COP_hp_cooling,
             d.hp_model, d.HP_size) = Heat_Pump_Model(
                d.T, d.P / 10, d.Hload, d.Cload)
            d.Eload_hp = eload_hp.to_numpy() if hasattr(eload_hp, 'to_numpy') else np.array(eload_hp)
        elif brand == 'Bosch':
            from sama.models.BB_HP_Bosch import Heat_Pump_Model
            (eload_hp, d.power_hp_heating, d.power_hp_cooling,
             d.COP_hp_heating, d.COP_hp_cooling,
             d.hp_model, d.HP_size) = Heat_Pump_Model(
                d.T, d.P / 10, d.Hload, d.Cload)
            d.Eload_hp = np.array(eload_hp)
        else:
            d.Eload_hp = 0
    except Exception as e:
        print(f"  ⚠️  HP load recalc failed: {e}")


# ─────────────────────────────────────────────────────────────────────────────
# Load rebuild
# ─────────────────────────────────────────────────────────────────────────────

def _recalc_thermal_load(d, config):
    """Reload Hload and Cload from house_load.xlsx (Tload_type == 1)."""
    tload_type = int(config.get('Tload_type', getattr(d, 'Tload_type', 1)))
    if tload_type == 1:
        try:
            import pandas as pd
            from sama import get_content_path
            fname = config.get('path_house_load', 'house_load.xlsx')
            try:
                p = get_content_path(fname)
            except Exception:
                p = fname
            data = pd.read_excel(p)
            d.Hload = data.iloc[:, 1].to_numpy()
            d.Cload = data.iloc[:, 2].to_numpy()
            d.Tload = d.Hload + d.Cload
            print(f"  → Thermal load reloaded from {p}")
        except Exception as e:
            print(f"  ⚠️  Thermal load reload failed: {e}")


def _recalc_eload(d, config):
    load_type = config.get('load_type')
    if load_type is None:
        _reattach_hp_load(d)
        return
    try:
        if load_type == 1:
            import pandas as pd
            from sama import get_content_path
            try:
                p = get_content_path(config.get('path_Eload', 'Eload.csv'))
            except Exception:
                p = config.get('path_Eload', 'Eload.csv')
            d.Eload_eh = np.array(pd.read_csv(p, header=None).values[:, 0])
        elif load_type == 2:
            from sama.utilities.dataextender import dataextender
            d.Eload_eh = dataextender(d.daysInMonth,
                         _to_np_array(config['Monthly_haverage_load']))
        elif load_type == 3:
            from sama.utilities.dataextender import dataextender
            d.Eload_eh = dataextender(d.daysInMonth,
                         _to_np_array(config['Monthly_daverage_load']) / 24)
        elif load_type == 4:
            from sama.utilities.dataextender import dataextender
            t = _to_np_array(config['Monthly_total_load'])
            d.Eload_eh = dataextender(d.daysInMonth, t / (d.daysInMonth * 24))
        elif load_type == 5:
            from sama.utilities.generic_load import generic_load
            d.Eload_eh = generic_load(5, 1, config.get('peak_month', 'July'),
                         d.daysInMonth, _to_np_array(config['user_defined_load']))
        elif load_type == 6:
            d.Eload_eh = np.full(8760, float(config['Annual_haverage_load']))
        elif load_type == 7:
            d.Eload_eh = np.full(8760, float(config['Annual_daverage_load']) / 24)
        elif load_type == 8:
            from sama.utilities.generic_load import generic_load
            d.Eload_eh = generic_load(8, 1, config.get('peak_month', 'July'),
                         d.daysInMonth, float(config['Annual_total_load']))
        elif load_type == 9:
            from sama.utilities.generic_load import generic_load
            d.Eload_eh = generic_load(9, 1, config.get('peak_month', 'July'),
                         d.daysInMonth, 1)
        elif load_type == 10:
            import pandas as pd
            from sama import get_content_path
            from sama.utilities.generic_load import generic_load
            try:
                p = get_content_path(config.get('path_Eload_daily', 'Eload_daily.csv'))
            except Exception:
                p = config.get('path_Eload_daily', 'Eload_daily.csv')
            user_load = np.array(pd.read_csv(p, header=None).values[:, 0])
            d.Eload_eh = generic_load(10, 1, config.get('peak_month', 'July'),
                         d.daysInMonth, user_load)
    except Exception as e:
        print(f"  ⚠️  Eload_eh rebuild failed (load_type={load_type}): {e}")
        return

    _reattach_hp_load(d)


def _reattach_hp_load(d):
    """
    Mirror Input_Data logic: Eload = Eload_eh + Eload_hp.
    Also keep both Eload_Previous / EloadPrevious spellings in sync.
    """
    eload_hp = getattr(d, 'Eload_hp', 0)
    if getattr(d, 'HP', 0) == 1 and eload_hp is not None and not (
            isinstance(eload_hp, int) and eload_hp == 0):
        try:
            hp = (eload_hp.to_numpy()
                  if hasattr(eload_hp, 'to_numpy') else np.array(eload_hp))
            d.Eload          = d.Eload_eh + hp
            d.Eload_Previous = d.Eload_eh + hp
            d.EloadPrevious  = d.Eload_Previous
        except Exception as e:
            print(f"  ⚠️  HP load reattach failed: {e}")
            d.Eload = d.Eload_eh
            d.Eload_Previous = d.Eload_eh
            d.EloadPrevious  = d.Eload_eh
    else:
        d.Eload = d.Eload_eh
        d.Eload_Previous = getattr(d, 'Eload_Previous', d.Eload_eh)
        d.EloadPrevious  = d.Eload_Previous

def _recalc_eload_previous(d, config):
    """
    Rebuild Eload_Previous from config when load_previous_year_type != 1.
    Mirrors the full InData logic for types 2-11.
    """
    ltype = int(config.get('load_previous_year_type',
                getattr(d, 'load_previous_year_type', 1)))
    try:
        if ltype == 1:
            # Same as current year — already set by _reattach_hp_load
            pass
        elif ltype == 2:
            import pandas as pd
            from sama import get_content_path
            try:
                p = get_content_path(config.get('path_Eload_Previous', 'Eload_previousyear.csv'))
            except Exception:
                p = config.get('path_Eload_Previous', 'Eload_previousyear.csv')
            d.Eload_Previous = np.array(pd.read_csv(p, header=None).values[:, 0])
            d.EloadPrevious  = d.Eload_Previous
        elif ltype == 3:
            from sama.utilities.dataextender import dataextender
            arr = _to_np_array(config['Monthly_haverage_load_previous'])
            d.Eload_Previous = dataextender(d.daysInMonth, arr)
            d.EloadPrevious  = d.Eload_Previous
        elif ltype == 4:
            from sama.utilities.dataextender import dataextender
            arr = _to_np_array(config['Monthly_daverage_load_previous']) / 24
            d.Eload_Previous = dataextender(d.daysInMonth, arr)
            d.EloadPrevious  = d.Eload_Previous
        elif ltype == 5:
            from sama.utilities.dataextender import dataextender
            arr = _to_np_array(config['Monthly_total_load_previous'])
            havg = arr / (d.daysInMonth * 24)
            d.Eload_Previous = dataextender(d.daysInMonth, havg)
            d.EloadPrevious  = d.Eload_Previous
        elif ltype == 6:
            from sama.utilities.generic_load import generic_load
            user_load = _to_np_array(config.get('Monthly_total_load_previous',
                                     config.get('user_defined_load_previous',
                                     [300]*12)))
            d.Eload_Previous = generic_load(1, ltype,
                               config.get('peak_month', 'July'),
                               d.daysInMonth, user_load)
            d.EloadPrevious  = d.Eload_Previous
        elif ltype == 7:
            val = float(config.get('Annual_haverage_load_previous', 1))
            d.Eload_Previous = np.full(8760, val)
            d.EloadPrevious  = d.Eload_Previous
        elif ltype == 8:
            val = float(config.get('Annual_daverage_load_previous', 10)) / 24
            d.Eload_Previous = np.full(8760, val)
            d.EloadPrevious  = d.Eload_Previous
        elif ltype == 9:
            from sama.utilities.generic_load import generic_load
            val = float(config.get('Annual_total_load_previous', 9500))
            d.Eload_Previous = generic_load(1, ltype,
                               config.get('peak_month', 'July'),
                               d.daysInMonth, val)
            d.EloadPrevious  = d.Eload_Previous
        elif ltype == 10:
            from sama.utilities.generic_load import generic_load
            d.Eload_Previous = generic_load(1, ltype,
                               config.get('peak_month', 'July'),
                               d.daysInMonth, 1)
            d.EloadPrevious  = d.Eload_Previous
        elif ltype == 11:
            import pandas as pd
            from sama import get_content_path
            from sama.utilities.generic_load import generic_load
            try:
                p = get_content_path(config.get('path_Eload_Previous', 'Eload_daily.csv'))
            except Exception:
                p = config.get('path_Eload_Previous', 'Eload_daily.csv')
            user_load = np.array(pd.read_csv(p, header=None).values[:, 0])
            d.Eload_Previous = generic_load(10, 1,
                               config.get('peak_month', 'July'),
                               d.daysInMonth, user_load)
            d.EloadPrevious  = d.Eload_Previous
    except Exception as e:
        print(f"  ⚠️  Eload_Previous rebuild failed (type={ltype}): {e}")




# ─────────────────────────────────────────────────────────────────────────────
# Cbuy rebuild
#
# KEY DESIGN:
#  - Always reads season, hours, and prices from `config` (not from d.*)
#    to avoid stale RS8 values left by Input_Data.__init__.
#  - onHours / midHours / ultraLowHours are passed as numpy object arrays
#    where every element is a Python list — identical to Input_Data:
#        np.array([[16,17,18,19,20],[16,17,18,19,20]], dtype=object)
#  - For unequal-length inner lists (common in RS7 where summer ≠ winter
#    peak hours), _to_hours_object_array preserves each list independently.
# ─────────────────────────────────────────────────────────────────────────────

def _recalc_cbuy(d, config):
    rs = int(config.get('rateStructure', getattr(d, 'rateStructure', 1)))
    d.rateStructure = rs
    print(f"  → Rebuilding Cbuy (rateStructure={rs})...")

    try:
        if rs == 1:  # Flat rate
            from sama.pricing.calcFlatRate import calcFlatRate
            price       = float(config.get('flatPrice', getattr(d, 'flatPrice', 0.2)))
            d.flatPrice = price
            d.Cbuy      = calcFlatRate(price)

        elif rs == 2:  # Seasonal rate
            from sama.pricing.calcSeasonalRate import calcSeasonalRate
            prices           = _to_np_array(config['seasonalPrices'])
            season           = _to_np_array(config['season'], dtype=int)
            d.seasonalPrices = prices
            d.season         = season
            d.Cbuy           = calcSeasonalRate(prices, season, d.daysInMonth)

        elif rs == 3:  # Monthly rate
            from sama.pricing.calcMonthlyRate import calcMonthlyRate
            prices          = _to_np_array(config['monthlyPrices'])
            d.monthlyPrices = prices
            d.Cbuy          = calcMonthlyRate(prices, d.daysInMonth)

        elif rs == 4:  # Tiered rate
            from sama.pricing.calcTieredRate import calcTieredRate
            prices       = _to_np_array(config['tieredPrices'])
            tier_mx      = _to_np_array(config['tierMax'])
            d.tieredPrices = prices
            d.tierMax      = tier_mx
            d.Cbuy         = calcTieredRate(prices, tier_mx, d.Eload, d.daysInMonth)

        elif rs == 5:  # Seasonal tiered rate
            from sama.pricing.calcSeasonalTieredRate import calcSeasonalTieredRate
            prices  = np.array(config['seasonalTieredPrices'], dtype=float)
            tier_mx = np.array(config['seasonalTierMax'],      dtype=float)
            season  = _to_np_array(config['season'], dtype=int)
            d.seasonalTieredPrices = prices
            d.seasonalTierMax      = tier_mx
            d.season               = season
            d.Cbuy = calcSeasonalTieredRate(prices, tier_mx, d.Eload, season)

        elif rs == 6:  # Monthly tiered rate
            from sama.pricing.calcMonthlyTieredRate import calcMonthlyTieredRate
            prices  = np.array(config['monthlyTieredPrices'], dtype=float)
            tier_mx = np.array(config['monthlyTierLimits'],   dtype=float)
            d.monthlyTieredPrices = prices
            d.monthlyTierLimits   = tier_mx
            d.Cbuy = calcMonthlyTieredRate(prices, tier_mx, d.Eload)

        elif rs == 7:  # Time of Use (TOU)
            from sama.pricing.calcTouRate import calcTouRate
            on_p   = _to_np_array(config['onPrice'])
            mid_p  = _to_np_array(config['midPrice'])
            off_p  = _to_np_array(config['offPrice'])
            season = _to_np_array(config['season'], dtype=int)
            treat  = bool(config.get('treat_special_days_as_offpeak', False))
            # Store on d.* as object arrays (preserves exact list-of-lists)
            on_h_obj  = _to_hours_object_array(config['onHours'])
            mid_h_obj = _to_hours_object_array(config['midHours'])
            # Pass to calcTouRate as equal-length 2D int arrays
            # (calcTouRate uses onHours[1, :] which requires rectangular array)
            on_h_2d  = _to_equal_hours_array(config['onHours'])
            mid_h_2d = _to_equal_hours_array(config['midHours'])
            d.onPrice  = on_p;  d.midPrice = mid_p;  d.offPrice = off_p
            d.onHours  = on_h_obj;  d.midHours = mid_h_obj;  d.season = season
            d.treat_special_days_as_offpeak = treat
            d.Cbuy = calcTouRate(
                d.year, on_p, mid_p, off_p,
                on_h_2d, mid_h_2d, season,
                d.daysInMonth, d.holidays, treat)

        elif rs == 8:  # Ultra-Low TOU (e.g. Ontario ULO)
            from sama.pricing.calcULTouRate import calcULTouRate
            on_p   = _to_np_array(config['onPrice'])
            mid_p  = _to_np_array(config['midPrice'])
            off_p  = _to_np_array(config['offPrice'])
            ulo_p  = _to_np_array(config['ultraLowPrice'])
            season = _to_np_array(config['season'], dtype=int)
            treat  = bool(config.get('treat_special_days_as_offpeak', True))
            on_h_obj  = _to_hours_object_array(config['onHours'])
            mid_h_obj = _to_hours_object_array(config['midHours'])
            ulo_h_obj = _to_hours_object_array(config['ultraLowHours'])
            on_h_2d   = _to_equal_hours_array(config['onHours'])
            mid_h_2d  = _to_equal_hours_array(config['midHours'])
            ulo_h_2d  = _to_equal_hours_array(config['ultraLowHours'])
            d.onPrice       = on_p;  d.midPrice = mid_p;  d.offPrice = off_p
            d.ultraLowPrice = ulo_p
            d.onHours       = on_h_obj;  d.midHours = mid_h_obj
            d.ultraLowHours = ulo_h_obj;  d.season  = season
            d.treat_special_days_as_offpeak = treat
            d.Cbuy = calcULTouRate(
                d.year, on_p, mid_p, off_p, ulo_p,
                on_h_2d, mid_h_2d, ulo_h_2d, season,
                d.daysInMonth, d.holidays, treat)

        else:
            print(f"  ⚠️  Unknown rateStructure={rs}")

    except Exception as e:
        import traceback
        print(f"  ⚠️  Cbuy rebuild failed (rateStructure={rs}): {e}")
        traceback.print_exc()


# ─────────────────────────────────────────────────────────────────────────────
# Csell rebuild
# NOTE: sellStructure==3 uses d.Cbuy — must be called AFTER _recalc_cbuy.
# ─────────────────────────────────────────────────────────────────────────────

def _recalc_csell(d, config):
    ss = int(config.get('sellStructure', getattr(d, 'sellStructure', 2)))
    d.sellStructure = ss
    print(f"  → Rebuilding Csell (sellStructure={ss})...")
    try:
        if ss == 1:  # Flat rate
            price   = float(config.get('Csell_flat', 0.1))
            d.Csell = np.full(8760, price)
        elif ss == 2:  # Monthly rate
            from sama.pricing.calcMonthlyRate import calcMonthlyRate
            prices  = _to_np_array(config['monthlysellprices'])
            d.monthlysellprices = prices
            d.Csell = calcMonthlyRate(prices, d.daysInMonth)
        elif ss == 3:  # Same as buy rate — uses already-rebuilt d.Cbuy
            d.Csell = d.Cbuy
    except Exception as e:
        print(f"  ⚠️  Csell rebuild failed (sellStructure={ss}): {e}")


# ─────────────────────────────────────────────────────────────────────────────
# Cbuy_NG rebuild
# The wizard saves ALL prices in the user's native unit ($/m³, $/kWh, or $/therm).
# NG_unit in config tells us which. This function is the SINGLE point where
# conversion to $/kWh happens — never anywhere else.
# ─────────────────────────────────────────────────────────────────────────────

def _recalc_cbuy_ng(d, config):
    rs_ng = int(config.get('rateStructure_NG', getattr(d, 'rateStructure_NG', 1)))
    d.rateStructure_NG = rs_ng
    print(f"  → Rebuilding Cbuy_NG (rateStructure_NG={rs_ng})...")

    ng_ec  = float(config.get('NG_energycontent', getattr(d, 'NG_energycontent', 10.81)))
    f_eff  = float(config.get('Furnace_eff',      getattr(d, 'Furnace_eff',      0.94)))
    ng_unit = config.get('NG_unit', 'm3')   # 'm3' | 'kwh' | 'therms'

    # ── Unit conversion helpers ───────────────────────────────────────────────
    # price: $/unit → $/kWh
    def to_kwh_price(v):
        v = np.array(v, dtype=float) if hasattr(v, '__len__') else float(v)
        if ng_unit == 'm3':
            return v / (ng_ec * f_eff)      # $/m³ → $/kWh
        elif ng_unit == 'therms':
            return v / 29.3001              # $/therm → $/kWh
        else:                               # already $/kWh
            return v

    # volume: native unit → kWh  (for tier limits)
    def to_kwh_vol(v):
        v = np.array(v, dtype=float) if hasattr(v, '__len__') else float(v)
        if ng_unit == 'm3':
            return v * ng_ec               # m³ → kWh
        elif ng_unit == 'therms':
            return v * 29.3001             # therms → kWh
        else:
            return v                       # already kWh

    # Grid_Tax_amount_NG and Grid_credit_NG: stored in native unit → convert
    if 'Grid_Tax_amount_NG' in config:
        d.Grid_Tax_amount_NG = to_kwh_price(float(config['Grid_Tax_amount_NG']))
    if 'Grid_credit_NG' in config:
        # credit is $/unit (same as Gas Tax Amount) — must convert to $/kWh
        d.Grid_credit_NG = to_kwh_price(float(config['Grid_credit_NG']))

    try:
        if rs_ng == 1:  # Flat rate
            from sama.pricing.calcFlatRate import calcFlatRate
            d.flatPrice_NG = to_kwh_price(float(config.get('flatPrice_NG', 0.28)))
            d.Cbuy_NG = calcFlatRate(d.flatPrice_NG)

        elif rs_ng == 2:  # Seasonal rate
            from sama.pricing.calcSeasonalRate import calcSeasonalRate
            prices    = to_kwh_price(_to_np_array(config['seasonalPrices_NG']))
            season_ng = _to_np_array(config['season_NG'], dtype=int)
            d.seasonalPrices_NG = prices
            d.season_NG         = season_ng
            d.Cbuy_NG = calcSeasonalRate(prices, season_ng, d.daysInMonth)

        elif rs_ng == 3:  # Monthly rate
            from sama.pricing.calcMonthlyRate import calcMonthlyRate
            prices = to_kwh_price(_to_np_array(config['monthlyPrices_NG']))
            d.monthlyPrices_NG = prices
            d.Cbuy_NG = calcMonthlyRate(prices, d.daysInMonth)

        elif rs_ng == 4:  # Tiered rate
            from sama.pricing.calcTieredRate import calcTieredRate
            prices  = to_kwh_price(_to_np_array(config['tieredPrices_NG']))
            tier_mx = to_kwh_vol(_to_np_array(config['tierMax_NG']))
            tload   = getattr(d, 'Tload', getattr(d, 'Hload', d.Eload) + getattr(d, 'Cload', 0))
            d.tieredPrices_NG = prices
            d.tierMax_NG      = tier_mx
            d.Cbuy_NG = calcTieredRate(prices, tier_mx, tload, d.daysInMonth)

        elif rs_ng == 5:  # Seasonal tiered rate
            from sama.pricing.calcSeasonalTieredRate import calcSeasonalTieredRate
            prices    = to_kwh_price(np.array(config['seasonalTieredPrices_NG'], dtype=float))
            tier_mx   = to_kwh_vol(np.array(config['seasonalTierMax_NG'], dtype=float))
            season_ng = _to_np_array(config['season_NG'], dtype=int)
            tload     = getattr(d, 'Tload', getattr(d, 'Hload', d.Eload) + getattr(d, 'Cload', 0))
            d.seasonalTieredPrices_NG = prices
            d.seasonalTierMax_NG      = tier_mx
            d.season_NG               = season_ng
            d.Cbuy_NG = calcSeasonalTieredRate(prices, tier_mx, tload, season_ng)

        elif rs_ng == 6:  # Monthly tiered rate
            from sama.pricing.calcMonthlyTieredRate import calcMonthlyTieredRate
            prices  = to_kwh_price(np.array(config['monthlyTieredPrices_NG'], dtype=float))
            tier_mx = to_kwh_vol(np.array(config['monthlyTierLimits_NG'], dtype=float))
            tload   = getattr(d, 'Tload', getattr(d, 'Hload', d.Eload) + getattr(d, 'Cload', 0))
            d.monthlyTieredPrices_NG = prices
            d.monthlyTierLimits_NG   = tier_mx
            d.Cbuy_NG = calcMonthlyTieredRate(prices, tier_mx, tload)

        elif rs_ng == 7:  # Therms-based 2-tier (e.g. PG&E G-1)
            from sama.pricing.calcMonthlyTieredRate import calcMonthlyTieredRate
            kwh_per_therm = 29.3001
            base_rate_kwh   = float(config['ng7_base_rate'])   / kwh_per_therm
            excess_rate_kwh = float(config['ng7_excess_rate'])  / kwh_per_therm
            baseline_therms = np.array(config['ng7_baseline_therms'], dtype=float)
            baseline_kwh    = baseline_therms * kwh_per_therm
            d.monthlyTierLimits_NG   = np.column_stack([baseline_kwh, np.full(12, 999999), np.full(12, 999999)])
            d.monthlyTieredPrices_NG = np.column_stack([
                np.full(12, base_rate_kwh),
                np.full(12, excess_rate_kwh),
                np.full(12, excess_rate_kwh)
            ])
            tload = getattr(d, 'Tload', getattr(d, 'Hload', d.Eload) + getattr(d, 'Cload', 0))
            d.Cbuy_NG = calcMonthlyTieredRate(d.monthlyTieredPrices_NG, d.monthlyTierLimits_NG, tload)

        elif rs_ng == 8:  # 4-tier m³-based (e.g. Enbridge EGD)
            from sama.pricing.calcMonthlyTieredRate4 import calcMonthlyTieredRate4
            prices_m3  = np.array(config['ng8_prices'], dtype=float)  # always $/m³
            limits_m3  = np.array(config['ng8_limits'], dtype=float)  # always m³
            # Always convert from m³ regardless of ng_unit (RS8 is always m³-native)
            prices_kwh = prices_m3 / (ng_ec * f_eff)
            limits_kwh = limits_m3 * ng_ec
            d.monthlyTierLimits_NG   = np.tile(np.append(limits_kwh, 999999), (12, 1))
            d.monthlyTieredPrices_NG = np.tile(prices_kwh, (12, 1))
            tload = getattr(d, 'Tload', getattr(d, 'Hload', d.Eload) + getattr(d, 'Cload', 0))
            d.Cbuy_NG = calcMonthlyTieredRate4(d.monthlyTieredPrices_NG, d.monthlyTierLimits_NG, tload)

        else:
            print(f"  ⚠️  Unknown rateStructure_NG={rs_ng}")

    except Exception as e:
        import traceback
        print(f"  ⚠️  Cbuy_NG rebuild failed (rateStructure_NG={rs_ng}): {e}")
        traceback.print_exc()


# ─────────────────────────────────────────────────────────────────────────────
# SCALAR FIELDS MAP
# (yaml_key) → (indata_attribute, python_type)
# ─────────────────────────────────────────────────────────────────────────────

_SCALAR_FIELDS = {
    # Optimization
    'Cash_Flow_adv':('Cash_Flow_adv',int), 'MaxIt':('MaxIt',int),
    'nPop':('nPop',int), 'Run_Time':('Run_Time',int),
    'w':('w',float), 'wdamp':('wdamp',float),
    'c1':('c1',float), 'c2':('c2',float),
    'F_min':('F_min',float), 'F_max':('F_max',float),
    'CR_min':('CR_min',float), 'CR_max':('CR_max',float),
    'maxTrials':('maxTrials',int),
    'modification_rate':('modification_rate',float),
    'initial_search_radius':('initial_search_radius',float),
    'final_search_radius':('final_search_radius',float),
    # Calendar
    'n':('n',int), 'year':('year',int),
    # Load flags / paths
    'load_type':('load_type',int),
    'load_previous_year_type':('load_previous_year_type',int),
    'Tload_type':('Tload_type',int),
    'G_type':('G_type',int), 'T_type':('T_type',int), 'WS_type':('WS_type',int),
    'azimuth':('azimuth',float), 'tilt':('tilt',float), 'soiling':('soiling',float),
    # Constraints — rates only; LPSP_max and RE_min are in _SKIP_KEYS (derived)
    'LPSP_max_rate':('LPSP_max_rate',float),
    'RE_min_rate':('RE_min_rate',float),
    'EM':('EM',int),
    # Component toggles
    'PV':('PV',int), 'WT':('WT',int), 'DG':('DG',int), 'Bat':('Bat',int),
    'Grid':('Grid',int), 'HP':('HP',int), 'EV':('EV',int),
    'Lead_acid':('Lead_acid',int), 'Li_ion':('Li_ion',int),
    'NEM':('NEM',int), 'NG_Grid':('NG_Grid',int),
    'cap_option':('cap_option',int), 'cap_size':('cap_size',float),
    'NEM_fee':('NEM_fee',float),
    'generation_cap':('generation_cap',float),
    'available_roof_surface':('available_roof_surface',float),
    'PVPanel_surface_per_rated_capacity':('PVPanel_surface_per_rated_capacity',float),
    # Financial
    'n_ir_rate':('n_ir_rate',float), 'n_ir':('n_ir',float),
    'e_ir_rate':('e_ir_rate',float), 'e_ir':('e_ir',float),
    'Budget':('Budget',float), 'Tax_rate':('Tax_rate',float),
    'RE_incentives_rate':('RE_incentives_rate',float),
    'Pricing_method':('Pricing_method',int),
    # PV
    'Total_PV_price':('Total_PV_price',float),
    'Engineering_Costs':('Engineering_Costs',float),
    'C_PV':('C_PV',float), 'R_PV':('R_PV',float), 'MO_PV':('MO_PV',float),
    'fpv':('fpv',float), 'Tcof':('Tcof',float), 'Tref':('Tref',float),
    'Tc_noct':('Tc_noct',float), 'Ta_noct':('Ta_noct',float),
    'G_noct':('G_noct',float), 'gama':('gama',float),
    'n_PV':('n_PV',float), 'Gref':('Gref',float),
    'L_PV':('L_PV',int), 'RT_PV':('RT_PV',int), 'Ppv_r':('Ppv_r',float),
    # Bottom-up engineering breakdown
    'Fieldwork':('Fieldwork',float), 'Officework':('Officework',float),
    'Other':('Other',float),
    'Permiting_and_Inspection':('Permiting_and_Inspection',float),
    'Electrical_BoS':('Electrical_BoS',float),
    'Structrual_BoS':('Structrual_BoS',float),
    'Supply_Chain_costs':('Supply_Chain_costs',float),
    'Profit_costs':('Profit_costs',float), 'Sales_tax':('Sales_tax',float),
    # Inverter
    'n_I':('n_I',float), 'DC_AC_ratio':('DC_AC_ratio',float),
    'L_I':('L_I',int), 'RT_I':('RT_I',int),
    'C_I':('C_I',float), 'R_I':('R_I',float), 'MO_I':('MO_I',float),
    # Wind turbine
    'C_WT':('C_WT',float), 'R_WT':('R_WT',float), 'MO_WT':('MO_WT',float),
    'L_WT':('L_WT',int), 'RT_WT':('RT_WT',int),
    'h_hub':('h_hub',float), 'h0':('h0',float), 'nw':('nw',float),
    'v_cut_in':('v_cut_in',float), 'v_rated':('v_rated',float),
    'v_cut_out':('v_cut_out',float),
    'alfa_wind_turbine':('alfa_wind_turbine',float), 'Pwt_r':('Pwt_r',float),
    # Diesel generator
    'C_DG':('C_DG',float), 'R_DG':('R_DG',float), 'MO_DG':('MO_DG',float),
    'TL_DG':('TL_DG',int), 'C_fuel':('C_fuel',float),
    'C_fuel_adj_rate':('C_fuel_adj_rate',float), 'C_fuel_adj':('C_fuel_adj',float),
    'a':('a',float), 'b':('b',float), 'LR_DG':('LR_DG',float),
    'CO2':('CO2',float), 'CO':('CO',float), 'NOx':('NOx',float), 'SO2':('SO2',float),
    'Cdg_r':('Cdg_r',float),
    # Battery common
    'C_B':('C_B',float), 'R_B':('R_B',float), 'MO_B':('MO_B',float),
    'SOC_min':('SOC_min',float), 'SOC_max':('SOC_max',float),
    'SOC_initial':('SOC_initial',float),
    'self_discharge_rate':('self_discharge_rate',float),
    'L_B':('L_B',int), 'RT_B':('RT_B',int),
    # Lead acid
    'Cnom_Leadacid':('Cnom_Leadacid',float),
    'alfa_battery_leadacid':('alfa_battery_leadacid',float),
    'c':('c',float), 'k_lead_acid':('k_lead_acid',float),
    'Ich_max_leadacid':('Ich_max_leadacid',float),
    'Vnom_leadacid':('Vnom_leadacid',float),
    'ef_bat_leadacid':('ef_bat_leadacid',float),
    'Q_lifetime_leadacid':('Q_lifetime_leadacid',float),
    # Li-ion
    'Cnom_Li':('Cnom_Li',float), 'Ich_max_Li_ion':('Ich_max_Li_ion',float),
    'Idch_max_Li_ion':('Idch_max_Li_ion',float),
    'alfa_battery_Li_ion':('alfa_battery_Li_ion',float),
    'Vnom_Li_ion':('Vnom_Li_ion',float), 'ef_bat_Li':('ef_bat_Li',float),
    'Q_lifetime_Li':('Q_lifetime_Li',float), 'Cbt_r':('Cbt_r',float),
    # Charger
    'C_CH':('C_CH',float), 'R_CH':('R_CH',float), 'MO_CH':('MO_CH',float),
    'L_CH':('L_CH',int), 'RT_CH':('RT_CH',int),
    # Heat pump
    'HP_brand':('HP_brand',str),
    'C_HP':('C_HP',float), 'R_HP':('R_HP',float), 'MO_HP':('MO_HP',float),
    'L_HP':('L_HP',int), 'RT_HP':('RT_HP',int), 'Php_r':('Php_r',float),
    # EV
    'C_ev':('C_ev',float), 'SOCe_min':('SOCe_min',float),
    'SOCe_max':('SOCe_max',float), 'SOCe_initial':('SOCe_initial',float),
    'Pev_max':('Pev_max',float), 'Range_EV':('Range_EV',float),
    'Daily_trip':('Daily_trip',float), 'SOC_dep':('SOC_dep',float),
    'n_e':('n_e',float),
    'self_discharge_rate_ev':('self_discharge_rate_ev',float),
    'L_EV_dis':('L_EV_dis',float),
    'degradation_percent':('degradation_percent',float),
    'step_km':('step_km',float),
    'L_EV':('L_EV',int), 'RT_EV':('RT_EV',int),
    'treat_special_days_as_home':('treat_special_days_as_home',bool),
    'Cost_EV':('Cost_EV',float), 'R_EVB':('R_EVB',float), 'MO_EV':('MO_EV',float),
    'Tin':('Tin',int), 'Tout':('Tout',int),
    # Grid / electricity scalars
    'Annual_expenses':('Annual_expenses',float),
    'Grid_sale_tax_rate':('Grid_sale_tax_rate',float),
    'Grid_Tax_amount':('Grid_Tax_amount',float),
    'Grid_credit':('Grid_credit',float),
    'Grid_escalation_projection':('Grid_escalation_projection',int),
    'Monthly_fixed_charge_system':('Monthly_fixed_charge_system',int),
    'SC_flat':('SC_flat',float),
    'SC_1':('SC_1',float), 'Limit_SC_1':('Limit_SC_1',float),
    'SC_2':('SC_2',float), 'Limit_SC_2':('Limit_SC_2',float),
    'SC_3':('SC_3',float), 'Limit_SC_3':('Limit_SC_3',float),
    'SC_4':('SC_4',float),
    # Rate selectors
    'rateStructure':('rateStructure',int),
    'sellStructure':('sellStructure',int),
    # Simple price scalars (some rate structures)
    'flatPrice':('flatPrice',float),
    'Csell_flat':('Csell_flat',float),
    'treat_special_days_as_offpeak':('treat_special_days_as_offpeak',bool),
    # Grid emissions
    'E_CO2':('E_CO2',float), 'E_SO2':('E_SO2',float), 'E_NOx':('E_NOx',float),
    'Pbuy_max':('Pbuy_max',float), 'Psell_max':('Psell_max',float),
    # NG scalars (unit-sensitive ones handled in _recalc_cbuy_ng → kept in _SKIP_KEYS)
    'NG_energycontent':('NG_energycontent',float),
    'Furnace_eff':('Furnace_eff',float),
    'rateStructure_NG':('rateStructure_NG',int),
    'Annual_expenses_NG':('Annual_expenses_NG',float),
    'Grid_sale_tax_rate_NG':('Grid_sale_tax_rate_NG',float),
    # Grid_credit_NG → SKIP_KEYS, set in _recalc_cbuy_ng (unit-converted)
    'Grid_escalation_projection_NG':('Grid_escalation_projection_NG',int),
    'Monthly_fixed_charge_system_NG':('Monthly_fixed_charge_system_NG',int),
    'SC_flat_NG':('SC_flat_NG',float),
}

# 1-D numpy array fields
_ARRAY_FIELDS = {
    'VarMin':('VarMin',float), 'VarMax':('VarMax',float),
    'holidays':('holidays',int),
    'Grid_escalation_rate':('Grid_escalation_rate',float),
    'Grid_escalation_rate_NG':('Grid_escalation_rate_NG',float),
    # Price arrays for non-TOU rate structures
    'onPrice':('onPrice',float), 'midPrice':('midPrice',float),
    'offPrice':('offPrice',float), 'ultraLowPrice':('ultraLowPrice',float),
    'seasonalPrices':('seasonalPrices',float),
    'monthlyPrices':('monthlyPrices',float),
    'tieredPrices':('tieredPrices',float), 'tierMax':('tierMax',float),
    'monthlysellprices':('monthlysellprices',float),
    'season':('season',int),
    'seasonalPrices_NG':('seasonalPrices_NG',float),
    'monthlyPrices_NG':('monthlyPrices_NG',float),
    'tieredPrices_NG':('tieredPrices_NG',float),
    'tierMax_NG':('tierMax_NG',float),
    'season_NG':('season_NG',int),
}

# Jagged / nested arrays — stored via _to_hours_object_array
# NOTE: onHours, midHours, ultraLowHours are NOT in _ARRAY_FIELDS to avoid
#       the flat-1D-array problem with unequal inner list lengths.
_OBJECT_ARRAY_FIELDS = {
    'onHours':'onHours', 'midHours':'midHours', 'ultraLowHours':'ultraLowHours',
    'seasonalTieredPrices':'seasonalTieredPrices',
    'seasonalTierMax':'seasonalTierMax',
    'monthlyTieredPrices':'monthlyTieredPrices',
    'monthlyTierLimits':'monthlyTierLimits',
    'seasonalTieredPrices_NG':'seasonalTieredPrices_NG',
    'seasonalTierMax_NG':'seasonalTierMax_NG',
    'monthlyTieredPrices_NG':'monthlyTieredPrices_NG',
    'monthlyTierLimits_NG':'monthlyTierLimits_NG',
}

# Keys whose YAML values must NOT be applied raw — derived or unit-converted
_SKIP_KEYS = {
    'input_directory', 'output_directory', 'optimization_algorithm',
    # Always recalculated from primary inputs
    'ir', 'System_Tax', 'RE_incentives',
    'LPSP_max', 'RE_min',
    'Grid_Tax', 'Grid_Tax_NG',        # derived from Grid_sale_tax_rate*/100
    'C_ev_usable', 'SOC_arr',
    'Grid_escalation', 'Grid_escalation_NG',  # rebuilt from rate arrays /100
    # Unit-converted in _recalc_cbuy_ng only (native unit → $/kWh)
    'flatPrice_NG', 'Grid_Tax_amount_NG', 'Grid_credit_NG',
    # Metadata — not InData attributes
    'NG_unit', 'compare_with_grid',
    # RS7/RS8 raw inputs — consumed entirely by _recalc_cbuy_ng
    'ng7_base_rate', 'ng7_excess_rate', 'ng7_baseline_therms',
    'ng8_prices', 'ng8_limits',
    'Service_charge_NG',   # set exclusively by _recalc_service_charge_ng
}


# ─────────────────────────────────────────────────────────────────────────────
# MAIN PUBLIC FUNCTION
# ─────────────────────────────────────────────────────────────────────────────

def apply_config(config):
    """
    Apply a wizard YAML config dict to the InData singleton.
    Returns the fully reconfigured InData instance.

    Strict execution order:
      1.  Scalars
      2.  1-D numpy arrays
      3.  Object arrays (onHours, midHours, nested price matrices)
      4.  Derived recalculations (dependency-ordered):
            year         → daysInMonth
            financial    → ir, System_Tax, RE_incentives
            constraints  → LPSP_max, RE_min
            grid/NG tax  → Grid_Tax, Grid_Tax_NG
            NEM invariant → NEM=0 when Grid=0
            pricing method (top-down recalculation)
            replacement times (all RT_*)
            battery rated capacity (Cbt_r)
            EV derived   → C_ev_usable, SOC_arr, Q_lifetime_ev, EV_p
            grid escalation → Grid_escalation, Grid_escalation_NG
            weather      → G, T, Vw
            HP load      → Eload_hp  (needs T, P from weather)
            electrical load → Eload_eh → Eload  (HP load added here)
            service charges (electricity + NG)
            Cbuy  ← ALWAYS rebuilt; reads ALL params from config (not d.*)
            Csell ← ALWAYS rebuilt; sellStructure==3 uses already-rebuilt Cbuy
            Cbuy_NG ← converts $/m³ → $/kWh exactly once
    """
    from sama.core.Input_Data import InData
    d = InData

    applied = 0
    issues  = []

    # ── Step 1: Scalars ───────────────────────────────────────────────────────
    for k, (attr, typ) in _SCALAR_FIELDS.items():
        if k in config and k not in _SKIP_KEYS:
            try:
                val = config[k]
                if typ == bool:
                    val = bool(val)
                elif val is not None:
                    val = typ(val)
                setattr(d, attr, val)
                applied += 1
            except Exception as e:
                issues.append(f"{k}: {e}")

    # ── Step 2: 1-D numpy arrays ──────────────────────────────────────────────
    for k, (attr, dtype) in _ARRAY_FIELDS.items():
        if k in config and k not in _SKIP_KEYS:
            try:
                setattr(d, attr, _to_np_array(config[k], dtype))
                applied += 1
            except Exception as e:
                issues.append(f"{k}: {e}")

    # (Service charge arrays handled by _recalc_service_charge_ng)

    # ── Step 3: Object arrays (hours, nested matrices) ────────────────────────
    for k, attr in _OBJECT_ARRAY_FIELDS.items():
        if k in config:
            try:
                setattr(d, attr, _to_hours_object_array(config[k]))
                applied += 1
            except Exception as e:
                issues.append(f"{k}: {e}")

    print(f"✅ Applied {applied} parameters from config")
    if issues:
        print(f"  ⚠️  {len(issues)} issue(s):")
        for s in issues[:15]:
            print(f"     • {s}")

    # ── Step 4: Derived recalculations (strict dependency order) ──────────────
    print("🔄 Recalculating derived parameters...")

    if 'year' in config:
        _recalc_days_in_month(d)

    _recalc_financial(d)
    _recalc_constraints(d)
    _recalc_grid_tax(d)
    _recalc_nem_invariant(d)            # enforce NEM=0 when Grid=0
    if 'compare_with_grid' in config:
        d.compare_with_grid = int(config['compare_with_grid'])
    _recalc_pricing_method(d, config)
    _recalc_replacement_times(d)
    _recalc_battery_rated_capacity(d)
    _recalc_ev_derived(d)

    if any(k in config for k in ('C_ev','degradation_percent','L_EV_dis',
                                  'Range_EV','step_km')):
        _recalc_ev_battery_throughput(d)

    if any(k in config for k in ('year','Tout','Tin','holidays',
                                  'treat_special_days_as_home')):
        _recalc_ev_presence(d)

    _recalc_grid_escalation(d)

    # Weather arrays (G, T, Vw) — also needed before HP load
    if any(k in config for k in ('G_type','T_type','WS_type','weather_url',
                                  'tilt','azimuth','soiling',
                                  'Monthly_average_temperature',
                                  'Annual_average_temperature',
                                  'Monthly_average_windspeed',
                                  'Annual_average_windspeed')):
        print("  → Rebuilding weather arrays (G, T, Vw)...")
        _recalc_weather(d, config)

    # Thermal load (Hload/Cload) — reload if house_load.xlsx or Tload_type changed
    if 'path_house_load' in config or 'Tload_type' in config:
        print("  → Rebuilding thermal load (Hload/Cload)...")
        _recalc_thermal_load(d, config)

    # HP load (needs T, P from weather AND updated Hload/Cload — runs after thermal reload)
    if any(k in config for k in ('HP','HP_brand','G_type','T_type','weather_url',
                                  'tilt','azimuth','soiling',
                                  'path_house_load','Tload_type')):
        if getattr(d, 'HP', 0) == 1:
            print("  → Rebuilding HP electrical load (Eload_hp)...")
            _recalc_hp_load(d, config)

    # Electrical load (Eload_eh → Eload, adds HP contribution)
    if any(k in config for k in ('load_type','path_Eload','Monthly_haverage_load',
                                  'Monthly_daverage_load','Monthly_total_load',
                                  'Annual_haverage_load','Annual_daverage_load',
                                  'Annual_total_load','path_Eload_daily','peak_month')):
        print("  → Rebuilding electrical load (Eload_eh → Eload)...")
        _recalc_eload(d, config)
    else:
        _reattach_hp_load(d)

    _recalc_service_charge(d)
    _recalc_service_charge_ng(d)

    # Rebuild Eload_Previous when type != 1 or specific array keys present
    if (('load_previous_year_type' in config and int(config['load_previous_year_type']) != 1)
            or any(k in config for k in ('Monthly_haverage_load_previous',
                                          'Monthly_daverage_load_previous',
                                          'Monthly_total_load_previous',
                                          'path_Eload_Previous'))):
        _recalc_eload_previous(d, config)

    # ── Cbuy — ALWAYS rebuild when rateStructure is present ──────────────────
    # This is the primary fix: _recalc_cbuy runs unconditionally so that
    # every rate structure change is guaranteed to update d.Cbuy.
    if 'rateStructure' in config:
        _recalc_cbuy(d, config)
    elif getattr(d, 'Grid', 0) == 0 and getattr(d, 'HP', 0) == 0:
        pass  # No grid and no HP — Cbuy unused, skip silently
    else:
        print("  ⚠️  rateStructure not in config — Cbuy not rebuilt (electricity costs may be wrong)")

    # ── Csell — ALWAYS rebuild when sellStructure is present ─────────────────
    # Must run AFTER Cbuy (sellStructure==3 copies d.Cbuy).
    if 'sellStructure' in config:
        _recalc_csell(d, config)

    # ── Cbuy_NG — rebuild when rateStructure_NG or any NG price key present ──
    if 'rateStructure_NG' in config or any(k in config for k in (
            'flatPrice_NG', 'NG_energycontent', 'Furnace_eff',
            'Grid_Tax_amount_NG', 'seasonalPrices_NG',
            'monthlyPrices_NG', 'tieredPrices_NG')):
        _recalc_cbuy_ng(d, config)

    print("✅ InData fully reconfigured and ready for optimization.\n")
    return d
