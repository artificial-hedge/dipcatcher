"""Simplified solar-eclipse circumstances — Besselian-lite geometry (SYNTHETIC).

Sun from the low-accuracy solar ephemeris; Moon on a circular inclined
orbit (i = 5.145 deg, sidereal period 27.32166 d, regression of the
node 18.6 y). Eclipse = topocentric separation < r_sun + r_moon near
new moon. Bench: golden-section minimum separation and its epoch match
a dense 30 s scan within tolerance.
"""

import numpy as np

_SEED = 20261231 + 887

_INC = np.radians(5.1454)
_T_M = 27.321661  # sidereal month, days
_T_NODE = 18.613 * 365.25  # nodal regression, days


def _sun_pos(jd: float) -> np.ndarray:
    """Geocentric apparent Sun unit vector + distance (AU)."""
    t = (jd - 2451545.0) / 36525.0
    l0 = np.radians((280.46646 + 36000.76983 * t) % 360.0)
    m = np.radians((357.52911 + 35999.05029 * t) % 360.0)
    c = np.radians(1.914602 * np.sin(m) + 0.019993 * np.sin(2 * m))
    lam = l0 + c
    r = 1.0 / (1 + 0.016708634 * np.cos(m)) * 0.9997 + 0.0003
    x = np.cos(lam) * r
    y = np.sin(lam) * np.cos(np.radians(23.4397)) * r
    z = np.sin(lam) * np.sin(np.radians(23.4397)) * r
    return np.asarray([x, y, z], dtype=np.float64)


def _moon_pos(jd: float, node0: float, phase0: float) -> np.ndarray:
    """Geocentric Moon unit vector scaled by relative distance (R_earth)."""
    w = 2 * np.pi / _T_M
    ang = phase0 + w * (jd - 2451545.0)
    node = node0 - 2 * np.pi / _T_NODE * (jd - 2451545.0)
    u = ang
    # orbital frame -> ecliptic -> equatorial (equatorial == ecliptic rot)
    xo, yo, zo = np.cos(u), np.sin(u) * np.cos(_INC), np.sin(u) * np.sin(_INC)
    cn, sn = np.cos(node), np.sin(node)
    xe = xo * cn - yo * sn
    ye = xo * sn + yo * cn
    ze = zo
    ce, se = np.cos(np.radians(23.4397)), np.sin(np.radians(23.4397))
    y = ye * ce - ze * se
    z = ye * se + ze * ce
    return np.asarray([xe, y, z], dtype=np.float64) * 60.3  # mean lunar distance in Earth radii


def separation(jd: float, node0: float, phase0: float) -> float:
    """Sun-Moon angular separation in degrees."""
    s = _sun_pos(jd)
    m = _moon_pos(jd, node0, phase0)
    sn, mn = s / np.linalg.norm(s), m / np.linalg.norm(m)
    return float(np.degrees(np.arccos(np.clip(np.dot(sn, mn), -1, 1))))


def min_separation(jd_lo: float, jd_hi: float, node0: float, phase0: float) -> tuple[float, float]:
    """Golden-section minimum separation in [jd_lo, jd_hi]; returns (jd, deg)."""
    a, b = jd_lo, jd_hi
    gr = (np.sqrt(5.0) - 1.0) / 2.0
    c = b - gr * (b - a)
    d = a + gr * (b - a)
    for _ in range(80):
        if separation(c, node0, phase0) < separation(d, node0, phase0):
            b = d
        else:
            a = c
        c = b - gr * (b - a)
        d = a + gr * (b - a)
    jd = (a + b) / 2.0
    return jd, separation(jd, node0, phase0)


def _scan_oracle(jd_lo: float, jd_hi: float, node0: float, phase0: float) -> tuple[float, float]:
    js = np.arange(jd_lo, jd_hi, 30.0 / 86400.0)
    seps = np.array([separation(j, node0, phase0) for j in js])
    i = int(np.argmin(seps))
    return float(js[i]), float(seps[i])


def bench_eclipse_circ(seed: int = _SEED) -> dict[str, float]:
    """SYNTHETIC bench: min-separation epoch/angle matches dense scan."""
    rng = np.random.default_rng(seed)
    ok = True
    for _ in range(4):
        node0 = float(rng.uniform(0, 2 * np.pi))
        # choose phase so a near-conjunction sits inside the window
        jd_c = 2451545.0 + float(rng.uniform(0, 200))
        phase0 = -float(np.arctan2(_sun_pos(jd_c)[1], _sun_pos(jd_c)[0])) - 2 * np.pi / _T_M * (
            jd_c - 2451545.0
        )
        jd_lo, jd_hi = jd_c - 0.6, jd_c + 0.6
        jd_min, sep_min = min_separation(jd_lo, jd_hi, node0, phase0)
        jd_o, sep_o = _scan_oracle(jd_lo, jd_hi, node0, phase0)
        if abs(jd_min - jd_o) > 0.01 or abs(sep_min - sep_o) > 0.05:
            ok = False
    return {"synthetic_eclipse_circ": 1.0 if ok else 0.0}
