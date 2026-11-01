"""
PV array output for mono-facial and bi-facial modules, with one or more
sub-arrays.

A PV array can be split into sub-arrays with their own orientation and rear
geometry. Each sub-array k receives the share ``share_k`` of the total PV
capacity. Its effective irradiance is

    G_eff_k = G_front_k + bifaciality * (1 + shade_factor) * G_rear_k

with G_rear_k = 0 for mono-facial modules, and its output follows SAMA's PV
model (temperature-corrected efficiency with the NOCT cell temperature
equation, evaluated with G_eff_k). For one mono-facial array the result is
exactly SAMA's original PV equation.
"""

import numpy as np

MONO = 'Mono-facial'
BI = 'Bi-facial'


def pv_power(Pn_PV, G_list, G_rear_list, shares, bifaciality, shade_factor,
             T, fpv, Gref, Tcof, Tref, Tc_noct, Ta_noct, G_noct, n_PV, gama):
    """
    Hourly DC output of the PV array [kW].

    G_list, G_rear_list and shares hold one entry per sub-array. A rear entry
    of None means a mono-facial sub-array.
    """
    Ppv = None
    for G_front, G_rear, share in zip(G_list, G_rear_list, shares):
        Pn_PV_k = Pn_PV * share
        if G_rear is None:
            G = G_front
        else:
            G = G_front + bifaciality * (1 + shade_factor) * G_rear
        # Module temperature (K) and output, as in SAMA's PV model
        Tc = (T + 273.15 + (Tc_noct - Ta_noct) * (G / G_noct) * (1 - ((n_PV * (1 - (Tcof / 100) * (Tref + 273.15))) / gama))) / (1 + (Tc_noct - Ta_noct) * (G / G_noct) * (((Tcof / 100) * n_PV) / gama))
        P_k = fpv * Pn_PV_k * (G / Gref) * (1 + (Tcof / 100) * (Tc - 273.15 - Tref))
        Ppv = P_k if Ppv is None else Ppv + P_k
    return Ppv


def default_subarrays(d):
    """One sub-array with the main tilt and azimuth and the whole capacity."""
    return [{'tilt': float(d.tilt), 'azimuth': float(d.azimuth), 'share': 1.0,
             'gcr': float(d.pv_gcr), 'height': float(d.pv_height),
             'pitch': float(d.pv_pitch)}]


def _validated_subarrays(d):
    subs = d.PV_subarrays if getattr(d, 'PV_subarrays', None) else default_subarrays(d)
    out = []
    for i, s in enumerate(subs):
        try:
            sub = {'tilt': float(s['tilt']), 'azimuth': float(s['azimuth']),
                   'share': float(s['share']),
                   'gcr': float(s.get('gcr', d.pv_gcr)),
                   'height': float(s.get('height', d.pv_height)),
                   'pitch': float(s.get('pitch', d.pv_pitch))}
        except (KeyError, TypeError, ValueError) as e:
            raise ValueError(f"PV sub-array {i + 1} needs numeric tilt, azimuth and share: {e}")
        if sub['share'] < 0:
            raise ValueError(f"PV sub-array {i + 1} has a negative share")
        out.append(sub)
    total = sum(s['share'] for s in out)
    if abs(total - 1.0) > 1e-9:
        raise ValueError(f"PV sub-array shares must sum to 1, got {total}")
    return out


def build_pv_irradiance(d, weather_url, soiling, G_type=1):
    """
    Set the sub-array irradiance on ``d`` (the InData object):
    ``d.PV_shares``, ``d.G_list`` (front POA per sub-array) and ``d.G_rear_list``
    (rear POA per sub-array, None for mono-facial modules).

    The front irradiance of a sub-array with the main tilt and azimuth is
    ``d.G``; other orientations are computed with SAMA's POA model from the same
    weather file. With G_type 2 (POA from a CSV file) only one sub-array is
    allowed, since the file gives a single orientation.
    """
    module = getattr(d, 'PV_module_type', MONO)
    if module not in (MONO, BI):
        raise ValueError(f"PV_module_type must be '{MONO}' or '{BI}', got '{module}'")
    if module == BI:
        if not 0 <= float(d.bifaciality) <= 1:
            raise ValueError("bifaciality must be between 0 and 1")
        if not -1 <= float(d.shade_factor) <= 0:
            raise ValueError("shade_factor must be between -1 and 0")

    subs = _validated_subarrays(d)
    if G_type != 1 and len(subs) > 1:
        raise ValueError("Several PV sub-arrays need G_type 1 (POA computed from the weather file)")

    G_list, G_rear_list = [], []
    for sub in subs:
        same_orientation = (sub['tilt'] == float(d.tilt) and sub['azimuth'] == float(d.azimuth))
        if G_type != 1 or same_orientation:
            G_front = np.asarray(d.G, dtype=float)
        else:
            from samapy.utilities.sam_monofacial_poa import runSimulation
            G_front = np.asarray(runSimulation(weather_url, sub['tilt'], sub['azimuth'], soiling)[0], dtype=float)
        G_list.append(G_front)
        if module == BI:
            from samapy.models.bifacial_rear import compute_rear_poa
            G_rear_list.append(np.asarray(compute_rear_poa(
                weather_url, surface_tilt=sub['tilt'], surface_azimuth=sub['azimuth'],
                gcr=sub['gcr'], height=sub['height'], pitch=sub['pitch'],
                bifaciality=float(d.bifaciality)), dtype=float))
        else:
            G_rear_list.append(None)

    d.PV_shares = [s['share'] for s in subs]
    d.G_list = G_list
    d.G_rear_list = G_rear_list
