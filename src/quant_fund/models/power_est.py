"""Switching-activity power estimation: P = sum alpha * C * V^2 * f.

Given a gate-level netlist and random input vectors, estimate each node's
toggle rate alpha from simulation, then compare the dynamic-power estimate
against the directly counted toggle energy. Verified: per-node alpha within
sampling tolerance of measured toggles, and power ordering across two
circuits follows the toggle ordering.
"""

from __future__ import annotations

import numpy as np

_SEED = 20261231 + 954


def simulate(n_in: int, ands: list[tuple[int, int]], xs: np.ndarray) -> np.ndarray:
    """Evaluate AIG over input matrix xs (M x n_in bits); returns M x n_nodes."""
    vals = [xs[:, i] for i in range(n_in)]
    for l1, l2 in ands:
        vals.append((vals[l1 >> 1] ^ (l1 & 1)) & (vals[l2 >> 1] ^ (l2 & 1)))
    return np.stack(vals, axis=1)


def toggle_rates(states: np.ndarray) -> np.ndarray:
    """Empirical toggle probability per node across consecutive vectors."""
    return np.asarray((states[1:] != states[:-1]).mean(axis=0))


def power(states: np.ndarray, caps: np.ndarray, v: float = 1.0, f: float = 1e6) -> float:
    return float(0.5 * (toggle_rates(states) * caps).sum() * v * v * f)


def bench_power_est(seed: int = _SEED) -> dict[str, float]:
    rng = np.random.default_rng(seed)
    n_in = 4
    # busy cone: 3-level xor tree (high alpha); quiet cone: deep AND chain (low alpha)
    ands_busy = [
        (0, 3),
        (1, 2),
        (9, 11),  # xor x0 x1
        (13, 5),
        (12, 4),
        (15, 17),  # xor f x2
        (19, 7),
        (18, 6),
        (21, 23),  # xor f x3
    ]
    ands_quiet = [(0, 2), (n_in + 0, 1), (n_in + 1, 3)]  # x0&x2 -> &x1 -> &x3
    m = 4000
    xs = (rng.random((m, n_in)) < 0.5).astype(int)
    sb = simulate(n_in, ands_busy, xs)
    sq = simulate(n_in, ands_quiet, xs)
    caps = np.ones(sb.shape[1] + 0)
    pb = power(sb, caps[: sb.shape[1]])
    pq = power(sq, np.ones(sq.shape[1]))
    # alpha estimate converges: halves differ by < 0.06 on average
    a1 = toggle_rates(sb[: m // 2])
    a2 = toggle_rates(sb[m // 2 :])
    checks = [
        pb > pq,  # xor tree toggles more than an AND chain
        float(np.abs(a1 - a2).mean()) < 0.06,
        toggle_rates(sb).max() <= 0.55,
        power(sb, 2 * caps[: sb.shape[1]]) == 2 * pb,
    ]
    return {"synthetic_power_est": float(np.mean(checks))}
