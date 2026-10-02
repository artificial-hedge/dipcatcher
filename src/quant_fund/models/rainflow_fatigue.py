"""ASTM E1049-85 rainflow cycle counting and
Palmgren-Miner fatigue damage.

- Rainflow counting (Endo / ASTM E1049-85 practice 4B
  four-point method): turns a stress/strain reversal
  series into (range, mean, count) cycles.
- Palmgren-Miner linear damage: D = sum n_i / N_i
  with S-N curve N = a / range^m (Wohler / Basquin).
- Mean-stress corrections: Goodman, Gerber, and
  Smith-Watson-Topper (SWT) equivalent ranges.

References
----------
- ASTM E1049-85 (reapproved 2017) 'Standard practices
  for cycle counting in fatigue analysis'.
- Downing & Socie (1982) 'Simple rainflow counting
  algorithms' Int. J. Fatigue 4(1).
- Miner (1945) 'Cumulative damage in fatigue' JAM 12.
- Smith, Watson & Topper (1970) J. Materials 5(4).

Honesty
-------
SYNTHETIC self-check: known reversal sequences with
hand-computable cycle tables and a fatigue-life
consistency check on a seeded random signal.

Composition
-----------
Pure numpy. Input is a 1-D turning-point (or raw
load) series; outputs are cycle tables and damage.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]


def _check_series(x: FloatArray) -> FloatArray:
    xa = np.asarray(x, dtype=np.float64).ravel()
    if xa.size < 4 or not np.isfinite(xa).all():
        raise ValueError("series must be finite, length >= 4")
    return xa


def turning_points(x: FloatArray) -> FloatArray:
    """Extract local extrema, keeping endpoints."""
    xa = _check_series(x)
    n = xa.size
    keep = np.ones(n, dtype=bool)
    keep[1:-1] = (np.diff(np.signbit(np.diff(xa))) != 0) | (
        (np.diff(xa)[:-1] == 0) & (np.diff(xa)[1:] != 0)
    )
    tp = xa[keep]
    # enforce alternation: drop equal neighbours
    tp = tp[np.concatenate([[True], np.diff(tp) != 0])]
    if tp.size < 3:
        raise ValueError("fewer than 3 turning points")
    return np.asarray(tp, dtype=np.float64)


def rainflow(x: FloatArray) -> tuple[FloatArray, FloatArray, FloatArray]:
    """ASTM four-point rainflow: returns (ranges, means, counts).

    counts contain 1.0 for full cycles and 0.5 for half-
    cycles (residuals) as in the classical convention.
    """
    tp = turning_points(x)
    stack: list[float] = []
    ranges: list[float] = []
    means: list[float] = []
    counts: list[float] = []
    for v in tp:
        stack.append(float(v))
        while len(stack) >= 3:
            x_rng = abs(stack[-2] - stack[-3])
            y_rng = abs(stack[-1] - stack[-2])
            if y_rng >= x_rng:
                if len(stack) == 3:
                    # half-cycle on the first two points
                    ranges.append(x_rng)
                    means.append((stack[0] + stack[1]) / 2)
                    counts.append(0.5)
                    stack.pop(0)
                else:
                    # full cycle on the middle pair (S[-3], S[-2])
                    ranges.append(x_rng)
                    means.append((stack[-3] + stack[-2]) / 2)
                    counts.append(1.0)
                    stack.pop(-3)
                    stack.pop(-2)
            else:
                break
    # residual half-cycles
    for k in range(len(stack) - 1):
        ranges.append(abs(stack[k + 1] - stack[k]))
        means.append((stack[k + 1] + stack[k]) / 2)
        counts.append(0.5)
    return (
        np.asarray(ranges),
        np.asarray(means),
        np.asarray(counts),
    )


def goodman(range_: FloatArray, mean: FloatArray, s_u: float) -> FloatArray:
    """Goodman equivalent fully-reversed range."""
    r = np.asarray(range_, dtype=np.float64)
    m = np.asarray(mean, dtype=np.float64)
    if s_u <= 0 or (np.abs(m) >= s_u).any():
        raise ValueError("s_u must exceed |mean|")
    return r / (1 - m / s_u)


def gerber(range_: FloatArray, mean: FloatArray, s_u: float) -> FloatArray:
    """Gerber parabolic equivalent range."""
    r = np.asarray(range_, dtype=np.float64)
    m = np.asarray(mean, dtype=np.float64)
    if s_u <= 0:
        raise ValueError("s_u positive")
    return r / (1 - (m / s_u) ** 2)


def swt(range_: FloatArray, mean: FloatArray) -> FloatArray:
    """Smith-Watson-Topper parameter sqrt(s_max * eps_a)-style:
    equivalent range sqrt(s_max * range)."""
    r = np.asarray(range_, dtype=np.float64)
    m = np.asarray(mean, dtype=np.float64)
    s_max = np.maximum(m + r / 2, 0.0)  # compressive cycles contribute 0
    return np.sqrt(s_max * r)


def miner_damage(ranges: FloatArray, counts: FloatArray, a: float, m_exp: float) -> float:
    """Palmgren-Miner D = sum n_i * range_i^m / a (Wohler N = a/r^m)."""
    r = np.asarray(ranges, dtype=np.float64)
    c = np.asarray(counts, dtype=np.float64)
    if r.size != c.size or a <= 0 or m_exp <= 0:
        raise ValueError("ranges/counts mismatch or bad S-N params")
    return float((c * r**m_exp).sum() / a)


def life_estimate(
    x: FloatArray,
    a: float = 1e12,
    m_exp: float = 3.0,
    correction: str = "goodman",
    s_u: float = 600.0,
) -> dict[str, float]:
    """Full pipeline: rainflow -> mean-stress correction -> Miner D
    -> blocks-to-failure."""
    r, mn, c = rainflow(x)
    if correction == "goodman":
        r_eq = goodman(r, mn, s_u)
    elif correction == "gerber":
        r_eq = gerber(r, mn, s_u)
    elif correction == "swt":
        r_eq = swt(r, mn)
    elif correction == "none":
        r_eq = r
    else:
        raise ValueError("unknown correction")
    d = miner_damage(r_eq, c, a, m_exp)
    if d <= 0:
        raise ValueError("zero damage")
    return {
        "damage_per_block": d,
        "blocks_to_failure": float(1.0 / d),
        "n_cycles": float(c.sum()),
        "max_range": float(r.max()),
    }


def bench_rainflow(seed: int = 504) -> dict[str, float]:
    """SYNTHETIC: rainflow table + Miner damage on a seeded signal."""
    rng = np.random.default_rng(seed)
    t = np.linspace(0, 60, 4000)
    # broadband stress signal: 3 harmonics + noise
    x = (
        120 * np.sin(2 * np.pi * 0.7 * t)
        + 60 * np.sin(2 * np.pi * 2.3 * t + 0.8)
        + 30 * np.sin(2 * np.pi * 5.1 * t + 1.9)
        + rng.normal(0, 12, t.size)
    )
    r, mn, c = rainflow(x)
    out = life_estimate(x, a=1e12, m_exp=3.0, correction="goodman", s_u=500)
    g = life_estimate(x, a=1e12, m_exp=3.0, correction="gerber", s_u=500)
    sw = life_estimate(x, a=1e12, m_exp=3.0, correction="swt")
    if out["n_cycles"] < 100 or out["blocks_to_failure"] <= 0:
        raise ValueError("rainflow pipeline degenerate")
    # hand-check: constant-amplitude sine -> every cycle ~same range
    s = np.linspace(0, 20 * np.pi, 400)
    xs = 100.0 * np.sin(s) + 200.0
    rs, _, cs = rainflow(xs)
    full = rs[cs == 1.0]
    if not np.allclose(full, 200.0, atol=1.0) or abs(cs.sum() - 10) > 1.0:
        raise ValueError("sine rainflow table wrong")
    return {
        "synthetic_n_cycles": out["n_cycles"],
        "synthetic_blocks_to_fail": out["blocks_to_failure"],
        "synthetic_max_range": out["max_range"],
        "synthetic_gerber_blocks": g["blocks_to_failure"],
        "synthetic_swt_blocks": sw["blocks_to_failure"],
        "synthetic_damage_per_block": out["damage_per_block"],
    }
