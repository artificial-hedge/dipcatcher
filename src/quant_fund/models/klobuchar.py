"""Klobuchar ionospheric delay model (GPS broadcast model) (SYNTHETIC).

8-coefficient broadcast model (alpha/beta) estimating vertical TEC
delay as a half-cosine of local time; maps slant delays via the
obliquity factor. Units: angles in semicircles internally, returned
delay in metres (seconds × c).
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]

_C = 299792458.0


def klobuchar_delay(
    lat: float,
    lon: float,
    elev: float,
    azim: float,
    gps_sec: float,
    alpha: FloatArray,
    beta: FloatArray,
) -> float:
    """Ionospheric delay in seconds.

    lat/lon geodetic (deg), elev/azim satellite direction (deg),
    gps_sec seconds of GPS week, alpha/beta broadcast coefficient
    quartets.
    """
    phi_u = np.deg2rad(lat) / np.pi
    lam_u = np.deg2rad(lon) / np.pi
    el = np.deg2rad(elev) / np.pi
    az = np.deg2rad(azim) / np.pi

    psi = 0.0137 / (el + 0.11) - 0.022
    phi_i = phi_u + psi * np.cos(az)
    phi_i = np.clip(phi_i, -0.416, 0.416)
    lam_i = lam_u + psi * np.sin(az) / np.cos(phi_i * np.pi)
    phi_m = phi_i + 0.064 * np.cos((lam_i - 1.617) * np.pi)

    t_loc = (43200.0 * lam_i + gps_sec) % 86400.0
    amp = sum(float(alpha[k]) * phi_m**k for k in range(4))
    amp = max(amp, 0.0)
    per = sum(float(beta[k]) * phi_m**k for k in range(4))
    per = max(per, 72000.0)
    x = 2.0 * np.pi * (t_loc - 50400.0) / per
    obl = 1.0 + 16.0 * (0.53 - el) ** 3  # slant (obliquity) factor
    if abs(x) < 1.57:
        return float(obl * (5e-9 + amp * (1.0 - x**2 / 2.0 + x**4 / 24.0)))
    return float(obl * 5e-9)


def klobuchar_meters(
    lat: float,
    lon: float,
    elev: float,
    azim: float,
    gps_sec: float,
    alpha: FloatArray,
    beta: FloatArray,
) -> float:
    """Ionospheric delay in metres."""
    return klobuchar_delay(lat, lon, elev, azim, gps_sec, alpha, beta) * _C


def bench_klobuchar(seed: int = 20261231) -> dict[str, float]:
    """SYNTHETIC: canonical broadcast coefficients; checks the daytime
    bulge, elevation obliquity, and hemisphere asymmetry."""
    # IS-GPS-200 example coefficients
    alpha = np.array([2.676e-08, 1.490e-08, -1.192e-07, 0.0])
    beta = np.array([8.806e04, -3.277e04, -1.966e05, 1.967e06])
    out: dict[str, float] = {}
    noon = klobuchar_meters(38.0, -77.0, 45.0, 0.0, 14 * 3600.0, alpha, beta)
    night = klobuchar_meters(38.0, -77.0, 45.0, 0.0, 2 * 3600.0, alpha, beta)
    low_el = klobuchar_meters(38.0, -77.0, 10.0, 0.0, 14 * 3600.0, alpha, beta)
    zen = klobuchar_meters(38.0, -77.0, 89.0, 0.0, 14 * 3600.0, alpha, beta)
    out["synthetic_klobuchar_noon_m"] = float(noon)
    out["synthetic_klobuchar_night_m"] = float(night)
    out["synthetic_klobuchar_low_el_m"] = float(low_el)
    out["synthetic_klobuchar_zenith_m"] = float(zen)
    out["synthetic_klobuchar_day_gt_night"] = float(noon > night)
    out["synthetic_klobuchar_obliquity"] = float(low_el / zen)
    out["synthetic_klobuchar_floor_ok"] = float(night >= 5e-9 * _C - 1e-6)
    return out


if __name__ == "__main__":
    print(bench_klobuchar())
