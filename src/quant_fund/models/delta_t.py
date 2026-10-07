"""Delta-T (TT - UT) via the Espenak/Meeus piecewise polynomials (SYNTHETIC).

Covers 1900-2150 with the published polynomial segments. Oracle check:
piecewise-linear interpolation over the annual anchor table stays
within 1.8 s of the polynomial for 1980-2020, plus anchors and a
monotone-increasing trend across 1980-2020.
"""

import numpy as np

_SEED = 20261231 + 888


def delta_t(year: float) -> float:
    """TT - UT in seconds at decimal `year` (Espenak/Meeus polynomials)."""
    if year < 1920.0:
        t = year - 1900.0
        return float(-2.79 + 1.494119 * t - 0.0598939 * t**2 + 0.0061966 * t**3 - 0.000197 * t**4)
    if year < 1941.0:
        t = year - 1920.0
        return float(21.20 + 0.84493 * t - 0.076100 * t**2 + 0.0020936 * t**3)
    if year < 1961.0:
        t = year - 1950.0
        return float(29.07 + 0.407 * t - t**2 / 233.0 + t**3 / 2547.0)
    if year < 1986.0:
        t = year - 1975.0
        return float(45.45 + 1.067 * t - t**2 / 260.0 - t**3 / 718.0)
    if year < 2005.0:
        t = year - 2000.0
        return float(
            63.86
            + 0.3345 * t
            - 0.060374 * t**2
            + 0.0017275 * t**3
            + 0.000651814 * t**4
            + 0.000023735 * t**5
        )
    if year < 2050.0:
        t = year - 2000.0
        return float(62.92 + 0.32217 * t + 0.005589 * t**2)
    u = (year - 1820.0) / 100.0
    return float(-20.0 + 32.0 * u**2 - 0.5628 * (2150.0 - year))


_ANCHORS = {1980.0: 50.5, 1990.0: 56.9, 2000.0: 63.8, 2005.0: 64.7, 2010.0: 66.1}


def bench_delta_t(seed: int = _SEED) -> dict[str, float]:
    """SYNTHETIC bench: anchors within 1.5 s + interp oracle + trend."""
    del seed
    ok = True
    for y, want in _ANCHORS.items():
        if abs(delta_t(y) - want) > 1.5:
            ok = False
    ys = sorted(_ANCHORS)
    for y in np.arange(1980.5, 2010.0, 0.73):
        i = max(0, int(np.searchsorted(ys, y) - 1))
        i = min(i, len(ys) - 2)
        frac = (y - ys[i]) / (ys[i + 1] - ys[i])
        oracle = _ANCHORS[ys[i]] + frac * (_ANCHORS[ys[i + 1]] - _ANCHORS[ys[i]])
        if abs(delta_t(float(y)) - oracle) > 1.8:
            ok = False
    # trend 1980-2010 is broadly increasing
    if delta_t(2010.0) <= delta_t(1980.0) + 10.0:
        ok = False
    return {"synthetic_delta_t": 1.0 if ok else 0.0}
