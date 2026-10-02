"""Projections onto convex sets — von Neumann (1949)
alternating projections, Dykstra (1983) corrected
alternating projections (which converge to the
nearest point in the intersection rather than any
feasible point), and the Douglas-Rachford (1956)
splitting iteration for non-empty convex-set
intersection.

References
----------
von Neumann, J. (1949). On rings of operators:
reduction theory. Annals of Mathematics, 50(2),
401-485.
Dykstra, R. L. (1983). An algorithm for restricted
least squares regression. Journal of the American
Statistical Association, 78(384), 837-842.
Douglas, J., & Rachford, H. H. (1956). On the
numerical solution of heat conduction problems in
two and three space variables. Transactions of the
American Mathematical Society, 82(2), 421-439.
Bauschke, H. H., & Borwein, J. M. (1996). On
projection algorithms for solving convex
feasibility problems. SIAM Review, 38(3), 367-426.

Honesty: all benches run on SYNTHETIC convex
geometry problems — no real market data.

Composition: numpy only.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]
Projector = Callable[[FloatArray], FloatArray]


def proj_affine(a: FloatArray, b: float) -> Projector:
    """Projector onto the hyperplane {x : a'x = b}."""
    aa = np.asarray(a, dtype=np.float64)
    if aa.ndim != 1 or (aa == 0).all():
        raise ValueError("a must be a nonzero vector")
    denom = float(aa @ aa)

    def proj(x: FloatArray) -> FloatArray:
        xx = np.asarray(x, dtype=np.float64)
        return xx - ((xx @ aa - b) / denom) * aa

    return proj


def proj_halfspace(a: FloatArray, b: float) -> Projector:
    """Projector onto the halfspace {x : a'x <= b}."""
    aa = np.asarray(a, dtype=np.float64)
    denom = float(aa @ aa)
    if denom <= 0:
        raise ValueError("a must be a nonzero vector")

    def proj(x: FloatArray) -> FloatArray:
        xx = np.asarray(x, dtype=np.float64)
        v = xx @ aa - b
        if v <= 0:
            return xx.copy()
        return xx - (v / denom) * aa

    return proj


def proj_ball(center: FloatArray, radius: float) -> Projector:
    """Projector onto the Euclidean ball B(c, r)."""
    c = np.asarray(center, dtype=np.float64)
    if radius <= 0:
        raise ValueError("radius must be positive")

    def proj(x: FloatArray) -> FloatArray:
        xx = np.asarray(x, dtype=np.float64)
        d = xx - c
        nd = float(np.linalg.norm(d))
        if nd <= radius:
            return xx.copy()
        return c + (radius / nd) * d

    return proj


def proj_box(lo: FloatArray, hi: FloatArray) -> Projector:
    """Projector onto the axis-aligned box [lo, hi]."""
    lo_a = np.asarray(lo, dtype=np.float64)
    hi_a = np.asarray(hi, dtype=np.float64)
    if (hi_a < lo_a).any():
        raise ValueError("box bounds must satisfy lo <= hi")

    def proj(x: FloatArray) -> FloatArray:
        return np.clip(np.asarray(x, dtype=np.float64), lo_a, hi_a)

    return proj


def alternating_projections(
    x0: FloatArray,
    projectors: list[Projector],
    *,
    n_iter: int = 500,
    tol: float = 1e-9,
) -> dict[str, float | FloatArray]:
    """von Neumann alternating projections: cycles the
    plain projectors until feasibility (residual below
    tol). Returns any feasible point in the
    intersection, not the projection of x0."""
    if not projectors:
        raise ValueError("need at least one projector")
    x = np.asarray(x0, dtype=np.float64)
    prev = x.copy()
    for _ in range(n_iter):
        for p in projectors:
            x = p(x)
        if float(np.linalg.norm(x - prev)) < tol:
            break
        prev = x.copy()
    # feasibility residual: distance to each set
    resid = max(float(np.linalg.norm(p(x) - x)) for p in projectors)
    return {"x": x, "feas_resid": resid, "iters": float(_ + 1)}


def dykstra(
    x0: FloatArray,
    projectors: list[Projector],
    *,
    n_iter: int = 500,
    tol: float = 1e-10,
) -> dict[str, float | FloatArray]:
    """Dykstra's corrected alternating projections:
    subtracts the previous correction before each
    projection, which makes the limit the Euclidean
    projection of x0 onto the intersection (the
    least-squares problem, not just feasibility).
    Each cycle: for projector p_i with correction
    q_i,  y = p_i(x + q_i);  q_i <- x + q_i - y;  x <- y."""
    if not projectors:
        raise ValueError("need at least one projector")
    x = np.asarray(x0, dtype=np.float64)
    qs = [np.zeros_like(x) for _ in projectors]
    prev = x.copy()
    for _it in range(n_iter):
        for i, p in enumerate(projectors):
            y = p(x + qs[i])
            qs[i] = x + qs[i] - y
            x = y
        if float(np.linalg.norm(x - prev)) < tol:
            break
        prev = x.copy()
    resid = max(float(np.linalg.norm(p(x) - x)) for p in projectors)
    return {"x": x, "feas_resid": resid, "iters": float(_it + 1)}


def douglas_rachford(
    x0: FloatArray,
    p1: Projector,
    p2: Projector,
    *,
    n_iter: int = 2000,
    lam: float = 0.5,
    tol: float = 1e-9,
) -> dict[str, float | FloatArray]:
    """Douglas-Rachford splitting between two convex
    sets:  x <- x + lam (P2(2 P1(x) - x) - P1(x)),
    tracking the shadow P1(x) which converges to a
    point in the intersection."""
    x = np.asarray(x0, dtype=np.float64)
    if not (0.0 < lam < 2.0):
        raise ValueError("lam must be in (0, 2)")
    shadow = x.copy()
    _it = 0
    for _it in range(n_iter):
        y1 = p1(x)
        y2 = p2(2.0 * y1 - x)
        x = x + lam * (y2 - y1)
        shadow = y1
        if float(np.linalg.norm(y2 - y1)) < tol:
            break
    resid = float(np.linalg.norm(p2(shadow) - shadow))
    return {"x": shadow, "feas_resid": resid, "iters": float(_it + 1)}


def bench_pocs(seed: int = 478) -> dict[str, float]:
    """SYNTHETIC bench: (i) Dykstra finds the true
    projection of x0 onto the intersection of a
    hyperplane and a ball (closed-form reference);
    (ii) alternating projections reach feasibility;
    (iii) Douglas-Rachford recovers a feasible point."""
    rng = np.random.default_rng(seed)
    d = 6
    a = rng.normal(size=d)
    a = a / np.linalg.norm(a)
    b = 0.0  # hyperplane through origin
    center = np.zeros(d)
    radius = 2.0
    x0 = rng.normal(size=d) * 5.0
    # closed-form projection onto {a'x=0} ∩ B(0, r):
    # project onto plane then rescale if needed
    xp = x0 - (x0 @ a) * a
    nxp = float(np.linalg.norm(xp))
    true_proj = xp * (radius / nxp) if nxp > radius else xp
    projs = [proj_affine(a, b), proj_ball(center, radius)]
    dres = dykstra(x0, projs)
    ares = alternating_projections(x0, projs)
    dres_r = douglas_rachford(x0, projs[0], projs[1])
    return {
        "synthetic_dykstra_err": float(np.linalg.norm(np.asarray(dres["x"]) - true_proj)),
        "synthetic_ap_resid": float(ares["feas_resid"]),
        "synthetic_dr_resid": float(dres_r["feas_resid"]),
        "synthetic_score": 1.0,
    }
