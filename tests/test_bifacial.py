"""Bi-facial PV modules and PV sub-arrays (samapy.models.pv_model, bifacial_rear)."""
import types

import numpy as np
import pytest

from samapy import get_content_path
from samapy.core.Input_Data import InData
from samapy.models.pv_model import BI, MONO, build_pv_irradiance, pv_power

WEATHER = get_content_path('METEO.csv')
PV_ARGS = dict(fpv=InData.fpv, Gref=InData.Gref, Tcof=InData.Tcof, Tref=InData.Tref,
               Tc_noct=InData.Tc_noct, Ta_noct=InData.Ta_noct, G_noct=InData.G_noct,
               n_PV=InData.n_PV, gama=InData.gama)


def original_pv_equation(Pn_PV, G):
    """SAMA's PV equation for one mono-facial array."""
    T, a = InData.T, PV_ARGS
    Tc = (T + 273.15 + (a['Tc_noct'] - a['Ta_noct']) * (G / a['G_noct']) * (1 - ((a['n_PV'] * (1 - (a['Tcof'] / 100) * (a['Tref'] + 273.15))) / a['gama']))) / (1 + (a['Tc_noct'] - a['Ta_noct']) * (G / a['G_noct']) * (((a['Tcof'] / 100) * a['n_PV']) / a['gama']))
    return a['fpv'] * Pn_PV * (G / a['Gref']) * (1 + (a['Tcof'] / 100) * (Tc - 273.15 - a['Tref']))


def site(**kw):
    """Inputs for build_pv_irradiance, with the default PV inputs of InData."""
    d = types.SimpleNamespace(G=InData.G, tilt=InData.tilt, azimuth=InData.azimuth,
                              PV_module_type=MONO, bifaciality=0.7, shade_factor=-0.02,
                              pv_gcr=0.35, pv_height=1.5, pv_pitch=6.0, PV_subarrays=None)
    for k, v in kw.items():
        setattr(d, k, v)
    build_pv_irradiance(d, WEATHER, InData.soiling)
    return d


def array_output(d, Pn_PV=10.0):
    return pv_power(Pn_PV, d.G_list, d.G_rear_list, d.PV_shares, d.bifaciality, d.shade_factor,
                    InData.T, **PV_ARGS)


def test_default_is_one_monofacial_array():
    assert InData.PV_module_type == MONO
    assert InData.PV_shares == [1.0] and InData.G_rear_list == [None]
    assert InData.G_list[0] is InData.G


def test_monofacial_output_is_the_original_equation():
    np.testing.assert_array_equal(array_output(site()), original_pv_equation(10.0, InData.G))


def test_rear_irradiance_matches_pvlib():
    import pandas as pd
    import pvlib
    from pvlib.bifacial import infinite_sheds
    from samapy.models.bifacial_rear import compute_rear_poa

    rear = compute_rear_poa(WEATHER, surface_tilt=33, surface_azimuth=180, gcr=0.35, height=1.5, pitch=6.0)
    meta = pd.read_csv(WEATHER, nrows=1)
    data = pd.read_csv(WEATHER, header=2)
    idx = pd.DatetimeIndex(pd.to_datetime(data[["Year", "Month", "Day", "Hour", "Minute"]]))
    idx = idx.tz_localize(f"Etc/GMT{-int(meta['Time Zone'].iloc[0]):+d}")
    sp = pvlib.solarposition.get_solarposition(idx, float(meta["Latitude"].iloc[0]),
                                               float(meta["Longitude"].iloc[0]))
    ref = infinite_sheds.get_irradiance(33, 180, sp["apparent_zenith"].to_numpy(), sp["azimuth"].to_numpy(),
                                        0.35, 1.5, 6.0, data["GHI"].to_numpy(float),
                                        data["DHI"].to_numpy(float), data["DNI"].to_numpy(float),
                                        data["Surface Albedo"].to_numpy(float))
    np.testing.assert_allclose(rear, np.nan_to_num(np.asarray(ref["poa_back"], float)))
    assert rear.shape == (8760,) and np.all(rear >= 0)
    assert np.all(rear[data["GHI"].to_numpy() == 0] == 0), "no rear irradiance at night"


def test_bifacial_adds_rear_irradiance():
    mono, bi = site(), site(PV_module_type=BI)
    G_eff = bi.G_list[0] + 0.7 * (1 - 0.02) * bi.G_rear_list[0]
    np.testing.assert_array_equal(array_output(bi), original_pv_equation(10.0, G_eff))
    gain = array_output(bi).sum() / array_output(mono).sum() - 1
    rear_share = 0.7 * 0.98 * bi.G_rear_list[0].sum() / bi.G_list[0].sum()
    assert 0.02 < gain < 0.2, f"annual bifacial gain {gain:.3f} outside the expected range"
    # the gain follows the added irradiance, slightly reduced by the hotter cells
    assert gain == pytest.approx(rear_share, rel=0.15) and gain < rear_share


def test_subarrays_split_the_capacity():
    east_west = [{'tilt': 90, 'azimuth': 90, 'share': 0.5}, {'tilt': 90, 'azimuth': 270, 'share': 0.5}]
    d = site(PV_subarrays=east_west)
    east = site(PV_subarrays=[dict(east_west[0], share=1.0)])
    west = site(PV_subarrays=[dict(east_west[1], share=1.0)])
    np.testing.assert_allclose(array_output(d), 0.5 * array_output(east) + 0.5 * array_output(west))
    assert not np.allclose(east.G_list[0], west.G_list[0])


@pytest.mark.parametrize("bad", [
    dict(PV_module_type='Tri-facial'),
    dict(PV_module_type=BI, bifaciality=1.5),
    dict(PV_module_type=BI, shade_factor=0.1),
    dict(PV_subarrays=[{'tilt': 30, 'azimuth': 180, 'share': 0.6}]),
    dict(PV_subarrays=[{'tilt': 30, 'share': 1.0}]),
])
def test_invalid_pv_inputs_are_rejected(bad):
    with pytest.raises(ValueError):
        site(**bad)


def test_fitness_uses_bifacial_irradiance(monkeypatch):
    import samapy.core.Fitness as F

    X = np.array([12.0, 0.0, 2.0, 0.0, 8.0])
    mono = F.evaluate(X)
    bi = site(PV_module_type=BI)
    for k in ('PV_module_type', 'G_list', 'G_rear_list', 'PV_shares', 'bifaciality', 'shade_factor'):
        monkeypatch.setattr(InData, k, getattr(bi, k))
    more_pv = F.evaluate(X)
    # more PV energy for the same design: less grid purchase and a lower cost
    assert more_pv["NPC"] < mono["NPC"]
    assert more_pv["RE"] > mono["RE"]
