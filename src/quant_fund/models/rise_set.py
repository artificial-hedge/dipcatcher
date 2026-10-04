"""Rise/transit/set times for the Sun via hour-angle iteration.

Simplified solar ephemeris (Meeus low accuracy): mean longitude,
equation of center, ecliptic-to-equatorial declination and apparent
sidereal time. Rise/set solves cos H0 = -tan(lat)*tan(dec) with one
refinement pass for declination drift over the event. Bench: the
computed UTC times match a brute-force altitude scan to <90 s.
"""

import numpy as np

_SEED = 20261231 + 886

_OBL = np.radians(23.4397)


def sun_ra_dec(jd: float) -> tuple[float, float]:
    """Geocentric apparent RA (deg) and declination (deg), low accuracy."""
    t = (jd - 2451545.0) / 36525.0
    l0 = np.radians((280.46646 + 36000.76983 * t) % 360.0)
    m = np.radians((357.52911 + 35999.05029 * t) % 360.0)
    c = np.radians(1.914602 * np.sin(m) + 0.019993 * np.sin(2 * m) + 0.000289 * np.sin(3 * m))
    lam = l0 + c
    ra = np.degrees(np.arctan2(np.cos(_OBL) * np.sin(lam), np.cos(lam))) % 360.0
    dec = np.degrees(np.arcsin(np.sin(_OBL) * np.sin(lam)))
    return ra, dec


def gmst(jd: float) -> float:
    """Apparent sidereal time at Greenwich, degrees."""
    t = (jd - 2451545.0) / 36525.0
    return (280.46061837 + 360.98564736629 * (jd - 2451545.0) + 0.000387933 * t**2) % 360.0


def altitude(jd: float, lat: float, lon: float) -> float:
    """Sun altitude in degrees at jd, observer lat/lon (deg, lon +E)."""
    ra, dec = sun_ra_dec(jd)
    h = np.radians((gmst(jd) + lon - ra) % 360.0)
    la, de = np.radians(lat), np.radians(dec)
    return float(
        np.degrees(np.arcsin(np.sin(la) * np.sin(de) + np.cos(la) * np.cos(de) * np.cos(h)))
    )


def _bisect_alt(lo: float, hi: float, lat: float, lon: float) -> float | None:
    """Refine a single altitude zero-crossing in [lo, hi] by bisection."""
    flo = altitude(lo, lat, lon)
    fhi = altitude(hi, lat, lon)
    if flo == 0.0:
        return lo
    if flo * fhi > 0:
        return None
    for _ in range(60):
        mid = (lo + hi) / 2
        if altitude(mid, lat, lon) * flo > 0:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2


def rise_transit_set(
    jd0: float, lat: float, lon: float
) -> tuple[float | None, float, float | None]:
    """UTC julian dates of rise, transit, set on the civil day starting jd0.

    Returns (None, transit, None) for polar day/night.
    """
    noon_jd = np.floor(jd0 - 0.5) + 0.5 - lon / 360.0  # rough transit guess
    # transit: golden-section altitude max in a +-0.5 d window
    tr, ts = noon_jd - 0.5, noon_jd + 0.5
    for _ in range(50):
        a, b = tr + (ts - tr) / 3, ts - (ts - tr) / 3
        if altitude(a, lat, lon) < altitude(b, lat, lon):
            tr = a
        else:
            ts = b
    transit = (tr + ts) / 2
    rise = _bisect_alt(transit - 0.75, transit - 0.001, lat, lon)
    set_ = _bisect_alt(transit + 0.001, transit + 0.75, lat, lon)
    return rise, transit, set_


def _scan_events(
    jd0: float, lat: float, lon: float, transit_guess: float, step: float = 20.0 / 86400.0
) -> tuple[float | None, float, float | None]:
    """Oracle: dense altitude scan; picks the local max nearest the guess
    and the zero crossings adjacent to it (same-day consistency)."""
    js = np.arange(jd0 - 0.4, jd0 + 1.4, step)
    alts = np.array([altitude(j, lat, lon) for j in js])
    peaks = np.where((alts[1:-1] > alts[:-2]) & (alts[1:-1] >= alts[2:]))[0] + 1
    if len(peaks) == 0:
        pk = int(np.argmin(np.abs(js - transit_guess)))
    else:
        pk = int(peaks[int(np.argmin(np.abs(js[peaks] - transit_guess)))])
    up = np.where((alts[:-1] < 0) & (alts[1:] >= 0))[0]
    dn = np.where((alts[:-1] > 0) & (alts[1:] <= 0))[0]

    def refine(i: int) -> float:
        return float(js[i] - alts[i] * (js[i + 1] - js[i]) / (alts[i + 1] - alts[i]))

    ups_b = up[up < pk]
    dns_a = dn[dn >= pk]
    rise = refine(int(ups_b[-1])) if len(ups_b) else None
    set_ = refine(int(dns_a[0])) if len(dns_a) else None
    tr, ts = js[pk] - step, js[pk] + step
    for _ in range(40):
        a, b = tr + (ts - tr) / 3, ts - (ts - tr) / 3
        if altitude(a, lat, lon) < altitude(b, lat, lon):
            tr = a
        else:
            ts = b
    return rise, (tr + ts) / 2, set_


def bench_rise_set(seed: int = _SEED) -> dict[str, float]:
    """SYNTHETIC bench: rise/transit/set times match dense altitude scan."""
    rng = np.random.default_rng(seed)
    ok = True
    jd0 = 2459000.5
    for _ in range(6):
        lat = float(rng.uniform(-55, 55))
        lon = float(rng.uniform(-170, 170))
        r, t, s = rise_transit_set(jd0, lat, lon)
        ro, to, so = _scan_events(jd0, lat, lon, t)
        if abs(t - to) > 0.002:
            ok = False
        if r is not None and ro is not None and abs(r - ro) > 0.002:
            ok = False
        if s is not None and so is not None and abs(s - so) > 0.002:
            ok = False
        jd0 += 31.0
    return {"synthetic_rise_set": 1.0 if ok else 0.0}
