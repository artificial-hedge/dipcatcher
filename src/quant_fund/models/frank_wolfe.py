"""Frank-Wolfe / conditional gradient (FW 1956; Jaggi 2013):
projection-free quadratic minimization over the probability
simplex and the ℓ1-ball, plus a pairwise-step variant that
escapes zig-zag slowdown. Synthetic bench gates FW error vs
the closed-form projected solution."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]


def _simplex_lmo(g: FloatArray) -> FloatArray:
    """argmin_{s∈Δ} <g, s> → vertex at min component."""
    s = np.zeros_like(g)
    s[int(np.argmin(g))] = 1.0
    return np.asarray(s)


def fw_simplex(
    a: FloatArray,
    b: FloatArray,
    it: int = 500,
    tol: float = 1e-12,
) -> FloatArray:
    """min ½ x'Ax + b'x over the simplex; linear-step FW."""
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    x = np.zeros_like(b)
    x[0] = 1.0
    for k in range(2, it + 2):
        g = a @ x + b
        s = _simplex_lmo(g)
        d = s - x
        denom = float(d @ a @ d)
        gamma = min(1.0, max(0.0, -float(g @ d) / denom)) if denom > 0 else 2.0 / k
        x += gamma * d
        if abs(float(g @ d)) < tol:
            break
    return np.asarray(x)


def _l1_lmo(g: FloatArray, radius: float) -> FloatArray:
    """argmin_{‖s‖₁≤r} <g, s> → -r·sign(g_j)·e_j at |g|max."""
    s = np.zeros_like(g)
    j = int(np.argmax(np.abs(g)))
    s[j] = -radius * np.sign(g[j])
    return np.asarray(s)


def fw_l1(
    a: FloatArray,
    b: FloatArray,
    radius: float = 1.0,
    it: int = 500,
    tol: float = 1e-12,
) -> FloatArray:
    """min ½ x'Ax + b'x over ‖x‖₁ ≤ radius."""
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    x = np.zeros_like(b)
    for _ in range(it):
        g = a @ x + b
        s = _l1_lmo(g, radius)
        d = s - x
        gap = -float(g @ d)
        if gap < tol:
            break
        denom = float(d @ a @ d)
        gamma = min(1.0, max(0.0, gap / denom)) if denom > 0 else 1.0
        x += gamma * d
    return np.asarray(x)


def pfw_simplex(
    a: FloatArray,
    b: FloatArray,
    it: int = 400,
    tol: float = 1e-12,
) -> FloatArray:
    """Pairwise FW over the simplex: move mass between the
    LMO atom and the max-weight away atom (Lacoste-Julien)."""
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    x = np.zeros_like(b)
    x[0] = 1.0
    for _ in range(it):
        g = a @ x + b
        i_in = int(np.argmin(g))  # LMO direction
        active = np.where(x > 1e-12)[0]
        i_out = int(active[np.argmax(g[active])])  # away atom
        if i_in == i_out:
            break
        if g[i_out] - g[i_in] < tol:
            break
        d = np.zeros_like(x)
        d[i_in] = 1.0
        d[i_out] = -1.0
        denom = float(d @ a @ d)
        gamma = -float(g @ d) / denom if denom > 0 else 0.0
        gamma = min(x[i_out], max(0.0, gamma))
        if gamma <= 0:
            break
        x += gamma * d
    return np.asarray(x)


def _project_simplex(v: FloatArray) -> FloatArray:
    """Euclidean projection onto Δ (for oracle comparison)."""
    u = np.sort(v)[::-1]
    css = np.cumsum(u)
    rho = int(np.nonzero(u * np.arange(1, len(u) + 1) > (css - 1))[0][-1])
    theta = (css[rho] - 1) / (rho + 1)
    return np.asarray(np.maximum(v - theta, 0.0))


def bench_frank_wolfe(seed: int = 561) -> dict[str, float]:
    """SYNTHETIC: min ½‖Mx‖² − c'x over Δ and the ℓ1-ball;
    FW iterates must approach the projected-oracle solution,
    and pairwise FW must dominate vanilla FW in duality gap."""
    rng = np.random.default_rng(seed)
    out: dict[str, float] = {}
    n = 20
    c = rng.normal(0, 1, n)
    # (a) A = I: min ½‖x‖² − c'x over Δ ↔ project c onto Δ
    # exactly — exact oracle gate for both FW variants.
    a_iso = np.eye(n)
    x_or = _project_simplex(c)
    x_fw = fw_simplex(a_iso, -c, it=1500)
    out["synthetic_fw_simplex_err"] = float(np.linalg.norm(x_fw - x_or))
    x_pfw = pfw_simplex(a_iso, -c, it=300)
    out["synthetic_pfw_simplex_err"] = float(np.linalg.norm(x_pfw - x_or))
    if out["synthetic_fw_simplex_err"] > 0.05:
        raise ValueError(f"fw simplex off: {out['synthetic_fw_simplex_err']}")
    if out["synthetic_pfw_simplex_err"] > 0.02:
        raise ValueError(f"pfw simplex off: {out['synthetic_pfw_simplex_err']}")
    # (b) general PD A: pairwise FW must not do worse than
    # vanilla FW in objective value (zig-zag escape).
    m = rng.normal(0, 1, (n + 6, n))
    a_gen = m.T @ m + 0.1 * np.eye(n)

    def f(x: FloatArray) -> float:
        return float(0.5 * x @ a_gen @ x - c @ x)

    f_fw = f(fw_simplex(a_gen, -c, it=1500))
    f_pfw = f(pfw_simplex(a_gen, -c, it=1500))
    out["synthetic_fw_obj"] = f_fw
    out["synthetic_pfw_obj"] = f_pfw
    if f_pfw > f_fw + 1e-9:
        raise ValueError(f"pfw not better: {f_pfw} vs {f_fw}")
    # l1 ball: min ½‖x−z‖² over ‖x‖₁ ≤ 1 → soft-threshold oracle
    z = rng.normal(0, 1, n) * 3
    x_l1 = fw_l1(np.eye(n), -z, radius=1.0, it=2000)
    # oracle: l1-ball projection = soft-threshold at λ where
    # ‖soft(z,λ)‖₁ = 1, found by bisection.
    lo, hi = 0.0, float(np.abs(z).max())
    for _ in range(100):
        mid = 0.5 * (lo + hi)
        if np.maximum(np.abs(z) - mid, 0).sum() > 1.0:
            lo = mid
        else:
            hi = mid
    xs = np.sign(z) * np.maximum(np.abs(z) - hi, 0)
    out["synthetic_fw_l1_err"] = float(np.linalg.norm(x_l1 - xs))
    if out["synthetic_fw_l1_err"] > 0.05:
        raise ValueError(f"fw l1 off: {out['synthetic_fw_l1_err']}")
    return out
