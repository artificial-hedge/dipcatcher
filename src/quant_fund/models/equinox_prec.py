"""IAU-1976 equinox precession — rotation of mean equatorial coords.

Angles in arcsec (Lieske et al. 1977): zeta, z, theta of Julian
centuries T. Precession matrix P = R3(-z) R2(+theta) R3(-zeta).
Physical invariants checked: P orthogonal, P(0) = I, and the
precessed celestial pole keeps constant distance from the ecliptic
pole (mean obliquity preserved along the precession cone).
"""

import numpy as np

_SEED = 20261231 + 884

_EPS0 = np.radians(23.4392911)  # mean obliquity at J2000


def _angles(T: float) -> tuple[float, float, float]:
    z = (2306.2181 * T + 1.09468 * T**2 + 0.018203 * T**3) / 3600.0
    th = (2004.3109 * T - 0.42665 * T**2 - 0.041833 * T**3) / 3600.0
    ze = (2306.2181 * T + 0.30188 * T**2 + 0.017998 * T**3) / 3600.0
    return np.radians(ze), np.radians(th), np.radians(z)


def _r3(a: float) -> np.ndarray:
    c, s = float(np.cos(a)), float(np.sin(a))
    return np.asarray([[c, s, 0.0], [-s, c, 0.0], [0.0, 0.0, 1.0]], dtype=np.float64)


def _r2(a: float) -> np.ndarray:
    c, s = float(np.cos(a)), float(np.sin(a))
    return np.asarray([[c, 0.0, -s], [0.0, 1.0, 0.0], [s, 0.0, c]], dtype=np.float64)


def precession_matrix(T: float) -> np.ndarray:
    ze, th, z = _angles(T)
    return np.asarray(_r3(-z) @ _r2(th) @ _r3(-ze), dtype=np.float64)


def precess(r_eq: np.ndarray, T: float) -> np.ndarray:
    return np.asarray(np.asarray(r_eq, dtype=np.float64) @ precession_matrix(T).T)


def bench_equinox_prec(seed: int = _SEED) -> dict[str, float]:
    """SYNTHETIC bench: orthogonality, P(0)=I, pole keeps obliquity distance."""
    del seed
    p0 = precession_matrix(0.0)
    if np.linalg.norm(p0 - np.eye(3)) > 1e-12:
        return {"synthetic_equinox_prec": 0.0}
    ecl = np.array([0.0, -np.sin(_EPS0), np.cos(_EPS0)])  # ecliptic pole, equatorial J2000
    pole = np.array([0.0, 0.0, 1.0])
    ok = True
    for T in (-1.0, -0.25, 0.5, 1.3):
        P = precession_matrix(T)
        if abs(np.linalg.det(P) - 1.0) > 1e-10:
            ok = False
        if np.linalg.norm(P.T @ P - np.eye(3)) > 1e-10:
            ok = False
        pole_t = P @ pole
        # constant angular separation from the ecliptic pole = obliquity
        if abs(float(np.dot(pole_t, ecl)) - np.cos(_EPS0)) > 2e-4:
            ok = False
        # round-trip: precess forward then un-precess via inverse transpose
        rt = P.T @ pole_t
        if np.linalg.norm(rt - pole) > 1e-12:
            ok = False
    return {"synthetic_equinox_prec": 1.0 if ok else 0.0}
