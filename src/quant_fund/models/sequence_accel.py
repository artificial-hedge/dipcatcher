"""Sequence-acceleration canon.

Classical extrapolation transforms for slowly converging
scalar sequences:

- ``aitken_delta2`` — one-shot Δ² transform of a 3-term window.
- ``aitken_iterate`` — iterated Δ² (Steffensen-style sweep).
- ``shanks`` — k-th order Shanks transform via the ε-algorithm's
  closed form on 2k+1 terms.
- ``wynn_epsilon`` — full Wynn ε-table; returns the deepest even
  column entry.
- ``richardson`` — polynomial Richardson extrapolation given
  f(h) values at geometrically shrinking h.

Bench: Leibniz π-series partial sums + trapezoid-rule
extrapolation — honest acceleration factors vs the raw tails
(SYNTHETIC).
"""

from __future__ import annotations

import numpy as np

FloatArray = np.ndarray


def aitken_delta2(s0: float, s1: float, s2: float) -> float:
    d = s2 - 2.0 * s1 + s0
    return s0 - (s1 - s0) ** 2 / d if abs(d) > 1e-300 else s2


def aitken_iterate(seq: FloatArray) -> FloatArray:
    """One full Δ² sweep; shrinks the sequence by 2."""
    out = np.empty(len(seq) - 2)
    for i in range(len(out)):
        out[i] = aitken_delta2(float(seq[i]), float(seq[i + 1]), float(seq[i + 2]))
    return out


def shanks(seq: FloatArray) -> float:
    """Deepest Wynn ε_{2k} entry (equivalent to order-k Shanks)."""
    return wynn_epsilon(seq)


def wynn_epsilon(seq: FloatArray) -> float:
    eps = np.zeros((len(seq), len(seq) + 1))
    eps[:, 1] = seq
    best = float(seq[-1])
    cap = 100.0 * float(np.max(np.abs(seq))) + 1.0
    for k in range(2, len(seq) + 1):
        for i in range(len(seq) - k + 1):
            d = eps[i + 1, k - 1] - eps[i, k - 1]
            eps[i, k] = eps[i + 1, k - 2] + (1.0 / d if abs(d) > 1e-300 else 0.0)
        # estimates sit in odd columns (k = 2m+1 <-> epsilon_{2m});
        # degenerate denominators signal the series is exhausted.
        cand = eps[0, k]
        if k % 2 == 1 and np.isfinite(cand) and abs(cand) <= cap:
            best = float(cand)
        elif k % 2 == 1:
            break
    return best


def richardson(fh: FloatArray, hs: FloatArray, p: int = 2) -> float:
    """Neville-style Richardson extrapolation to h→0."""
    n = len(fh)
    t = np.zeros((n, n))
    t[:, 0] = fh
    for k in range(1, n):
        for i in range(n - k):
            ratio = (hs[i + k] / hs[i]) ** p
            t[i, k] = t[i + 1, k - 1] + (t[i + 1, k - 1] - t[i, k - 1]) / (1.0 / ratio - 1.0)
    return float(t[0, n - 1])


def bench_sequence_accel(seed: int = 0) -> dict[str, float]:
    _ = np.random.default_rng(seed)
    # Leibniz pi: partial sums S_n = 4 sum (-1)^k/(2k+1).
    n = 60
    terms = 4.0 * (-1.0) ** np.arange(n) / (2 * np.arange(n) + 1)
    s = np.cumsum(terms)
    raw_err = abs(s[-1] - np.pi)
    ait = aitken_iterate(aitken_iterate(s))
    err_ait = abs(float(ait[-1]) - np.pi)
    err_wynn = abs(wynn_epsilon(s) - np.pi)
    # Trapezoid rule for int_0^1 exp(-x^2) at h, h/2, ...
    f = lambda t: np.exp(-(t**2))  # noqa: E731
    exact = 0.746824132812427  # known value
    hs = 0.5 ** np.arange(1, 6)
    fh = np.empty(5)
    for i, h in enumerate(hs):
        xs = np.linspace(0.0, 1.0, int(round(1.0 / h)) + 1)
        fh[i] = h * (0.5 * f(xs[0]) + f(xs[1:-1]).sum() + 0.5 * f(xs[-1]))
    err_trap = abs(fh[-1] - exact)
    err_rich = abs(richardson(fh, hs, p=2) - exact)
    return {
        "synthetic_leibniz_raw_err": float(raw_err),
        "synthetic_aitken_err": float(err_ait),
        "synthetic_wynn_err": float(err_wynn),
        "synthetic_trap_err": float(err_trap),
        "synthetic_richardson_err": float(err_rich),
    }


__all__ = [
    "aitken_delta2",
    "aitken_iterate",
    "bench_sequence_accel",
    "richardson",
    "shanks",
    "wynn_epsilon",
]
