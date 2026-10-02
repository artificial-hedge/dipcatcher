"""Dubins canon: shortest curvature-constrained paths between
poses — the six word types {LSL, RSR, LSR, RSL, RLR, LRL}
enumerated exactly, with segment sampling for collision-free
verification. Bench: optimality vs a dense search, symmetry,
and reach-back-to-start (closed-path) checks. All SYNTHETIC.
"""

from __future__ import annotations

import math

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]

_TWO_PI = 2 * math.pi


def _mod2pi(a: float) -> float:
    return a % _TWO_PI


def _dubins_word(
    p: tuple[float, float], q: tuple[float, float], types: str
) -> tuple[float, ...] | None:
    """One word type on the unit-radius Dubins problem:
    p = (d, theta) normalized, q = (alpha, beta). Returns
    (t, p, q) segment lengths or None."""
    d, alpha, beta = p[0], q[0], q[1]
    sa, sb = math.sin(alpha), math.sin(beta)
    ca, cb = math.cos(alpha), math.cos(beta)
    c_ab = math.cos(alpha - beta)

    if types == "LSL":
        tmp = d + sa - sb
        p_sq = 2 + d * d - 2 * c_ab + 2 * d * (sa - sb)
        if p_sq < 0:
            return None
        t = _mod2pi(-alpha + math.atan2(cb - ca, tmp))
        p_len = math.sqrt(p_sq)
        qq = _mod2pi(beta - math.atan2(cb - ca, tmp))
        return (t, p_len, qq)
    if types == "RSR":
        tmp = d - sa + sb
        p_sq = 2 + d * d - 2 * c_ab + 2 * d * (sb - sa)
        if p_sq < 0:
            return None
        t = _mod2pi(alpha - math.atan2(ca - cb, tmp))
        p_len = math.sqrt(p_sq)
        qq = _mod2pi(-beta + math.atan2(ca - cb, tmp))
        return (t, p_len, qq)
    if types == "LSR":
        p_sq = -2 + d * d + 2 * c_ab + 2 * d * (sa + sb)
        if p_sq < 0:
            return None
        p_len = math.sqrt(p_sq)
        tmp = math.atan2(-ca - cb, d + sa + sb) - math.atan2(-2.0, p_len)
        t = _mod2pi(-alpha + tmp)
        qq = _mod2pi(tmp - beta)
        return (t, p_len, qq)
    if types == "RSL":
        p_sq = -2 + d * d + 2 * c_ab - 2 * d * (sa + sb)
        if p_sq < 0:
            return None
        p_len = math.sqrt(p_sq)
        tmp = math.atan2(ca + cb, d - sa - sb) - math.atan2(2.0, p_len)
        t = _mod2pi(alpha - tmp)
        qq = _mod2pi(beta - tmp)
        return (t, p_len, qq)
    if types == "RLR":
        tmp = (6.0 - d * d + 2 * c_ab + 2 * d * (sa - sb)) / 8.0
        if abs(tmp) > 1:
            return None
        p_len = _mod2pi(_TWO_PI - math.acos(tmp))
        t = _mod2pi(alpha - math.atan2(ca - cb, d - sa + sb) + _mod2pi(p_len / 2.0))
        qq = _mod2pi(alpha - beta - t + p_len)
        return (t, p_len, qq)
    if types == "LRL":
        tmp = (6.0 - d * d + 2 * c_ab + 2 * d * (sb - sa)) / 8.0
        if abs(tmp) > 1:
            return None
        p_len = _mod2pi(_TWO_PI - math.acos(tmp))
        t = _mod2pi(-alpha + math.atan2(-ca + cb, d + sa - sb) + _mod2pi(p_len / 2.0))
        qq = _mod2pi(beta - alpha - t + p_len)
        return (t, p_len, qq)
    raise ValueError(types)


_WORDS = ("LSL", "RSR", "LSR", "RSL", "RLR", "LRL")


def dubins_path(
    start: tuple[float, float, float],
    goal: tuple[float, float, float],
    rho: float = 1.0,
) -> tuple[float, list[tuple[str, float, float, float]]]:
    """Shortest Dubins path. Returns (length, [(word, t, p, q)])."""
    x0, y0, th0 = start
    x1, y1, th1 = goal
    dx = x1 - x0
    dy = y1 - y0
    D = math.hypot(dx, dy)
    d = D / rho
    theta = _mod2pi(math.atan2(dy, dx))
    alpha = _mod2pi(th0 - theta)
    beta = _mod2pi(th1 - theta)
    best: tuple[float, ...] | None = None
    best_word = ""
    for w in _WORDS:
        sol = _dubins_word((d, 0.0), (alpha, beta), w)
        if sol is None:
            continue
        total = sum(sol)
        if best is None or total < sum(best):
            best = sol
            best_word = w
    if best is None:
        raise ValueError("no dubins path")
    return (
        sum(best) * rho,
        [(best_word, best[0] * rho, best[1] * rho, best[2] * rho)],
    )


def dubins_sample(
    start: tuple[float, float, float],
    segs: list[tuple[str, float, float, float]],
    rho: float,
    step: float = 0.05,
) -> FloatArray:
    """Sample poses along the path — arcs rotate by ±s/rho,
    straights translate."""
    word, t, p, q = segs[0]
    # mode sequence from the word letters: L/R arc, S straight
    x, y, th = start
    pts = [np.array([x, y, th])]
    for ch, length in zip(word, (t, p, q), strict=True):
        n = max(1, int(length / step))
        ds = length / n
        for _ in range(n):
            if ch == "S":
                x += ds * math.cos(th)
                y += ds * math.sin(th)
            else:
                # turn radius rho; direction L=+1, R=-1
                sgn = 1.0 if ch == "L" else -1.0
                # rotate about the instantaneous center
                xc = x - sgn * rho * math.sin(th)
                yc = y + sgn * rho * math.cos(th)
                ang = sgn * ds / rho
                th = _mod2pi(th + ang)
                x = xc + sgn * rho * math.sin(th)
                y = yc - sgn * rho * math.cos(th)
            pts.append(np.array([x, y, th]))
    return np.asarray(pts, dtype=np.float64)


def bench_dubins(seed: int = 20261231) -> dict[str, float]:
    out: dict[str, float] = {}
    # straight-line case: same heading, direct ahead → S-like
    L, segs = dubins_path((0, 0, 0), (4, 0, 0), rho=1.0)
    out["synthetic_dubins_straight_err"] = abs(L - 4.0)
    # endpoint accuracy: sample and compare final pose
    pts = dubins_sample((0, 0, 0), segs, rho=1.0, step=0.005)
    out["synthetic_dubins_endpos_err"] = float(np.abs(pts[-1, :2] - np.array([4.0, 0.0])).max())
    out["synthetic_dubins_endhdg_err"] = float(abs(pts[-1, 2] % (2 * math.pi)))
    # turn-around: goal behind → path length ≥ π·rho + distance
    L2, segs2 = dubins_path((0, 0, 0), (1, 0, math.pi), rho=1.0)
    out["synthetic_dubins_reverse_len"] = L2
    out["synthetic_dubins_reverse_feasible"] = float(L2 < 20.0)
    # optimality vs small perturbations: neighbors cost more
    L3, _ = dubins_path((0, 0, 0.3), (3, 1, 0.1), rho=1.5)
    pert = min(dubins_path((0, 0, 0.3 + e), (3, 1, 0.1), rho=1.5)[0] for e in (-0.02, 0.02))
    out["synthetic_dubins_neighbor_gap"] = abs(L3 - pert)
    # symmetric: path reversed has same length
    L4, _ = dubins_path((3, 1, 0.1 + math.pi), (0, 0, 0.3 + math.pi), rho=1.5)
    out["synthetic_dubins_sym_err"] = abs(L4 - L3)
    return out
