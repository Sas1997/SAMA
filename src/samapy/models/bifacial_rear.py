"""
bifacial_rear.py
----------------
Rear-side plane-of-array (POA) irradiance for a bifacial PV array, using
pvlib's infinite sheds view-factor model (pvlib.bifacial.infinite_sheds).

Rear irradiance per m^2 depends only on the weather, the sun position and the
fixed array geometry (tilt, azimuth, ground coverage ratio, height, pitch,
albedo). It does not depend on the PV capacity, so it is computed once when the
inputs are loaded. Inside the optimization only the effective irradiance is
formed (see samapy.models.pv_model):

    G_eff = G_front + bifaciality * (1 + shade_factor) * G_rear

which is the same combination pvlib uses for its ``poa_global`` output. The
front irradiance G_front is SAMA's existing plane-of-array irradiance; this
module only supplies the rear term. G_eff is also used in the cell temperature
equation, because nominal operating cell temperature coefficients are measured
for front-side irradiance only.
"""

import numpy as np
import pandas as pd


def compute_rear_poa(
    weather_url,
    surface_tilt=33.0,      # deg from horizontal (must match the front model)
    surface_azimuth=180.0,  # deg east of north: S=180, E=90, W=270 (must match front)
    gcr=0.35,               # ground coverage ratio = collector width / pitch
    height=1.5,             # row-center height above ground [same units as pitch]
    pitch=6.0,              # row-to-row spacing [same units as height]
    bifaciality=0.85,       # rear/front efficiency ratio; affects poa_global only
    return_global=False,    # also return poa_front and poa_global if True
):
    """
    Rear-side POA irradiance from an NSRDB-format weather file.

    The file must have the site metadata (Latitude, Longitude, Time Zone) on its
    first data row and the hourly columns (Year, Month, Day, Hour, Minute, GHI,
    DHI, DNI, Surface Albedo) under a header on its third row, as in SAMA's
    METEO.csv.

    Returns
    -------
    poa_back : np.ndarray, shape (8760,)
        Rear-side POA irradiance [W/m^2] before the bifaciality factor.
    (poa_back, poa_front, poa_global) if return_global is True.
    """
    try:
        import pvlib
        from pvlib.bifacial import infinite_sheds
    except ImportError as e:  # pragma: no cover - pvlib is a declared dependency
        raise ImportError("Bifacial PV modules need pvlib: pip install pvlib") from e

    # site coordinates and time zone from the metadata row
    meta = pd.read_csv(weather_url, nrows=1)
    lat = float(meta["Latitude"].iloc[0])
    lon = float(meta["Longitude"].iloc[0])
    tz = int(meta["Time Zone"].iloc[0])

    # hourly data block (column headers on row 3)
    data = pd.read_csv(weather_url, header=2)

    # time-zone-aware timestamps from the date and time columns
    idx = pd.to_datetime(data[["Year", "Month", "Day", "Hour", "Minute"]])
    idx = pd.DatetimeIndex(idx).tz_localize(f"Etc/GMT{-tz:+d}")  # tz=-5 -> 'Etc/GMT+5'

    ghi = data["GHI"].to_numpy(float)
    dhi = data["DHI"].to_numpy(float)
    dni = data["DNI"].to_numpy(float)
    albedo = data["Surface Albedo"].to_numpy(float)  # hourly, site-specific (NSRDB)

    # solar position (the file has the zenith only, the azimuth is also needed)
    solpos = pvlib.solarposition.get_solarposition(idx, lat, lon)
    solar_zenith = solpos["apparent_zenith"].to_numpy()
    solar_azimuth = solpos["azimuth"].to_numpy()

    res = infinite_sheds.get_irradiance(
        surface_tilt=surface_tilt,
        surface_azimuth=surface_azimuth,
        solar_zenith=solar_zenith,
        solar_azimuth=solar_azimuth,
        gcr=gcr,
        height=height,
        pitch=pitch,
        ghi=ghi,
        dhi=dhi,
        dni=dni,
        albedo=albedo,
        bifaciality=bifaciality,
    )

    poa_back = np.nan_to_num(np.asarray(res["poa_back"], dtype=float), nan=0.0)

    if return_global:
        poa_front = np.nan_to_num(np.asarray(res["poa_front"], float), nan=0.0)
        poa_global = np.nan_to_num(np.asarray(res["poa_global"], float), nan=0.0)
        return poa_back, poa_front, poa_global
    return poa_back
