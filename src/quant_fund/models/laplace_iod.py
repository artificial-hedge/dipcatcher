"""Laplace's method: initial orbit determination from three angles-only sightings.

SYNTHETIC bench: propagates a two-body orbit, samples topocentric
line-of-sight unit vectors at three epochs, recovers (r, v) at the middle
epoch and compares against truth.
"""

from __future__ import annotations

import numpy as np
from numpy.linalg import norm

_SEED = 20261231 + 920
_MU = 398600.4418  # km^3/s^2


def _accel(r: np.ndarray, mu: float) -> np.ndarray:
    rm = float(norm(r))
    return -mu * r / (rm**3)


def _rk4_step(r: np.ndarray, v: np.ndarray, dt: float, mu: float) -> tuple[np.ndarray, np.ndarray]:
    def f(rv: np.ndarray) -> np.ndarray:
        return np.concatenate([rv[3:], _accel(rv[:3], mu)])

    y = np.concatenate([r, v])
    k1 = f(y)
    k2 = f(y + dt / 2 * k1)
    k3 = f(y + dt / 2 * k2)
    k4 = f(y + dt * k3)
    yn = y + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
    return yn[:3], yn[3:]


def propagate(
    r: np.ndarray, v: np.ndarray, dt: float, mu: float = _MU, nstep: int = 20
) -> tuple[np.ndarray, np.ndarray]:
    for _ in range(max(1, nstep)):
        r, v = _rk4_step(r, v, dt / max(1, nstep), mu)
    return r, v


def los_to_radec(los: np.ndarray) -> tuple[float, float]:
    """Unit line-of-sight -> (ra, dec) radians."""
    ra = float(np.arctan2(los[1], los[0]))
    dec = float(np.arcsin(np.clip(los[2], -1.0, 1.0)))
    return ra, dec


def radec_to_los(ra: float, dec: float) -> np.ndarray:
    return np.asarray(
        [np.cos(dec) * np.cos(ra), np.cos(dec) * np.sin(ra), np.sin(dec)],
        dtype=np.float64,
    )


def laplace_iod(
    ras: np.ndarray,
    decs: np.ndarray,
    taus: tuple[float, float],
    site: np.ndarray,
    mu: float = _MU,
) -> tuple[np.ndarray, np.ndarray]:
    """Three angles-only sightings -> (r2, v2) at the middle epoch.

    `taus` = (t1-t2, t3-t2) in seconds (t1<t2<t3 so tau1<0<tau3).
    `site` = observer position (same units as orbit radius).
    Uses the classical Laplace construction with Lagrange f/g series
    truncated at the time-of-flight terms needed for short arcs.
    """
    tau1, tau3 = float(taus[0]), float(taus[1])
    los = np.asarray([radec_to_los(ra, d) for ra, d in zip(ras, decs, strict=True)])
    R = np.asarray(site, dtype=np.float64)

    # r_i = rho_i los_i + R and r_i ~ f_i r2 + g_i v2 (Lagrange f/g series)
    # =>  rho_i los_i + R = f_i (rho2 los2 + R) + g_i v2
    # unknowns (rho1, rho2, rho3, v2): 9 equations, iterated on r2 magnitude.
    def fg(tt: float, r2: float) -> tuple[float, float]:
        u = mu / r2**3
        f = 1.0 - 0.5 * u * tt * tt
        g = tt - u * tt**3 / 6.0
        return f, g

    def residual(x: np.ndarray) -> np.ndarray:
        """x = (rho2, vx, vy, vz); 6 LOS residuals at t1,t3."""
        rho2, v2 = x[0], x[1:4]
        r2 = R + rho2 * los[1]
        out = np.zeros(6)
        for k, (tau_i, li) in enumerate(((tau1, los[0]), (tau3, los[2]))):
            ri, _ = propagate(r2, v2, tau_i, mu, nstep=8)
            pred = ri - R
            pn = float(norm(pred))
            if pn <= 0 or rho2 <= 0:
                out[3 * k : 3 * k + 3] = 1e6
            else:
                out[3 * k : 3 * k + 3] = pred / pn - li
        return out

    def refine(x: np.ndarray) -> np.ndarray:
        for _ in range(40):
            r = residual(x)
            c = float(r @ r)
            if c < 1e-20:
                break
            J = np.zeros((6, 4))
            for j in range(4):
                h = max(1e-3, abs(x[j]) * 1e-6)
                xh = x.copy()
                xh[j] += h
                J[:, j] = (residual(xh) - r) / h
            try:
                dx = np.linalg.solve(J.T @ J + 1e-10 * np.eye(4), -J.T @ r)
            except np.linalg.LinAlgError:
                break
            # damped step
            lam = 1.0
            for _ls in range(10):
                xn = x + lam * dx
                if float(residual(xn) @ residual(xn)) < c:
                    x = xn
                    break
                lam *= 0.5
            else:
                break
        return x

    # multi-start over scalar rho2 with v2 seeded from the linear solve per guess
    best: tuple[float, np.ndarray] | None = None
    seed_vs: list[np.ndarray] = []
    r2m = float(norm(R)) + 1000.0
    A = np.zeros((6, 6))
    b = np.zeros(6)
    for _ in range(60):  # cheap linear seed for v2
        f1, g1 = fg(tau1, r2m)
        f3, g3 = fg(tau3, r2m)
        A[:3, 0] = los[0]
        A[:3, 1] = -f1 * los[1]
        A[:3, 3:6] = -g1 * np.eye(3)
        b[:3] = f1 * R - R
        A[3:6, 2] = los[2]
        A[3:6, 1] = -f3 * los[1]
        A[3:6, 3:6] = -g3 * np.eye(3)
        b[3:6] = f3 * R - R
        try:
            x = np.linalg.solve(A, b)
        except np.linalg.LinAlgError:
            break
        r2m_new = float(norm(x[1] * los[1] + R))
        seed_vs.append(x[3:6].copy())
        if abs(r2m_new - r2m) < 1e-9 * r2m:
            break
        r2m = r2m_new
    v_seed = seed_vs[-1] if seed_vs else np.zeros(3)
    for rho20 in np.geomspace(float(norm(R)) * 0.5, 8.0 * float(norm(R)), 30):
        for vv in (v_seed, np.zeros(3)):
            xf = refine(np.concatenate([[rho20], vv]))
            c = float(residual(xf) @ residual(xf))
            if best is None or c < best[0]:
                best = (c, xf)
    if not (best is not None):
        raise ValueError("best is not None")
    rho2f = best[1][0]
    if rho2f <= 0:
        raise RuntimeError("laplace_iod: no positive-range root")
    r2v = R + rho2f * los[1]
    v2 = best[1][1:4]
    return np.asarray(r2v, dtype=np.float64), np.asarray(v2, dtype=np.float64)


def bench_laplace_iod(seed: int = _SEED) -> dict[str, float]:
    rng = np.random.default_rng(seed)
    score = 0.0
    # truth orbit: circular LEO inclined
    r0 = np.asarray([7000.0, 0.0, 0.0])
    v0 = np.asarray([0.0, np.sqrt(_MU / 7000.0) * np.cos(0.3), np.sqrt(_MU / 7000.0) * np.sin(0.3)])
    site = np.asarray([0.0, 0.0, 6378.0])  # out of orbital plane -> well-posed
    # three epochs 20s apart around t2
    states = [propagate(r0, v0, dt) for dt in (-120.0, 0.0, 120.0)]
    ras, decs = [], []
    for r, _ in states:
        los = r - site
        los /= norm(los)
        ra, dec = los_to_radec(los)
        ras.append(ra)
        decs.append(dec)
    r2e, v2e = laplace_iod(np.asarray(ras), np.asarray(decs), (-120.0, 120.0), site)
    rerr = float(norm(r2e - states[1][0]) / norm(states[1][0]))
    verr = float(norm(v2e - states[1][1]) / norm(states[1][1]))
    score += 1.0 if rerr < 0.05 else 0.0
    score += 1.0 if verr < 0.15 else 0.0
    # recovered orbit propagates consistently with later LOS
    r3, _ = propagate(r2e, v2e, 120.0)
    los3 = r3 - site
    los3 /= norm(los3)
    ra3, d3 = los_to_radec(los3)
    score += 1.0 if abs(ra3 - ras[2]) < 1e-3 and abs(d3 - decs[2]) < 1e-3 else 0.0
    # sanity: semi-major axis of recovered state near 7000
    from quant_fund.models.orbital_elements import state_to_elements

    a, e, *_ = state_to_elements(r2e, v2e, _MU)
    score += 1.0 if abs(a - 7000.0) < 700.0 and e < 0.2 else 0.0
    rng.random()
    return {"synthetic_laplace_iod": score / 4.0}
