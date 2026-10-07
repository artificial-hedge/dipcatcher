"""Congestion potential game — best-response dynamics to Nash (SYNTHETIC).

N users each pick one of R routes; cost to a user on route r is
l_r(n_r) = a_r * n_r + b_r. Rosenthal potential
Phi = sum_r sum_{x=1..n_r} l_r(x) strictly decreases under unilateral
best responses, converging to an equilibrium. Bench: final potential,
Nash residual (max unilateral gain), and comparison vs random play.
"""

import numpy as np

from quant_fund.models._mfg_synth import POT_A, POT_B, POT_N


def _latency(n_r: np.ndarray) -> np.ndarray:
    return np.asarray(POT_A * n_r + POT_B, dtype=np.float64)


def _potential(n_r: np.ndarray) -> float:
    tot = 0.0
    for r, cnt in enumerate(n_r):
        for x in range(1, int(cnt) + 1):
            tot += POT_A[r] * x + POT_B[r]
    return float(tot)


def _nash_resid(n_r: np.ndarray) -> float:
    lat = _latency(n_r)
    worst = 0.0
    for r in range(len(n_r)):
        if n_r[r] == 0:
            continue
        for s in range(len(n_r)):
            if s != r:
                gain = lat[r] - (POT_A[s] * (n_r[s] + 1) + POT_B[s])
                worst = max(worst, gain)
    return float(worst)


def _br_dynamics(seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    counts = np.bincount(rng.integers(0, len(POT_A), POT_N), minlength=len(POT_A)).astype(float)
    for _ in range(400):
        r = int(rng.integers(0, len(POT_A)))
        if counts[r] <= 0:
            continue
        lat_now = POT_A[r] * counts[r] + POT_B[r]
        gains = [lat_now - (POT_A[s] * (counts[s] + 1) + POT_B[s]) for s in range(len(POT_A))]
        s = int(np.argmax(gains))
        if gains[s] <= 1e-12:
            continue
        counts[r] -= 1
        counts[s] += 1
    return counts


def bench_potential_game(seed: int = 4211) -> dict[str, float]:
    counts = _br_dynamics(seed)
    phi = _potential(counts)
    resid = _nash_resid(counts)
    rng = np.random.default_rng(seed + 7)
    rand_counts = np.bincount(rng.integers(0, len(POT_A), POT_N), minlength=len(POT_A))
    phi_rand = _potential(rand_counts.astype(float))
    social = float(np.sum(counts * _latency(counts)))
    return {
        "synthetic_pot_phi": phi,
        "synthetic_pot_nash_resid": resid,
        "synthetic_pot_phi_rand": phi_rand,
        "synthetic_pot_social": social,
    }
