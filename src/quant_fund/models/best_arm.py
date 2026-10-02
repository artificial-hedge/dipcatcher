"""Fixed-confidence best-arm identification canon:
successive elimination (Even-Dar, Mannor & Mansour 2006)
and LUCB-1 (Kalyanakrishnan, Tewari, Auer & Stone 2012),
validated on a synthetic Bernoulli instance with a known
optimal arm.
"""

from __future__ import annotations

import numpy as np

FloatArray = np.ndarray


def _ucb_bound(t: int, n: float, delta: float) -> float:
    return float(np.sqrt(np.log(4.0 * t * t / delta) / (2.0 * max(n, 1.0))))


def successive_elimination(
    probs: FloatArray,
    delta: float,
    rng: np.random.Generator,
    max_pulls: int = 20000,
) -> tuple[int, int, bool]:
    """Returns (identified arm, pulls used, correct)."""
    probs = np.asarray(probs, dtype=np.float64)
    k = probs.size
    best_true = int(np.argmax(probs))
    active = list(range(k))
    counts = np.zeros(k)
    means = np.zeros(k)
    t = 0
    while len(active) > 1 and t < max_pulls:
        for a in active:
            r = float(rng.random() < probs[a])
            counts[a] += 1.0
            means[a] += (r - means[a]) / counts[a]
        t += 1
        ucbs = {a: means[a] + _ucb_bound(t, counts[a], delta) for a in active}
        lcb_max = max(means[a] - _ucb_bound(t, counts[a], delta) for a in active)
        active = [a for a in active if ucbs[a] >= lcb_max]
    return int(active[0]), int(t * 1), bool(active[0] == best_true)


def lucb(
    probs: FloatArray,
    delta: float,
    rng: np.random.Generator,
    max_pulls: int = 20000,
) -> tuple[int, int, bool]:
    """LUCB-1: pull the empirical leader and its closest challenger."""
    probs = np.asarray(probs, dtype=np.float64)
    k = probs.size
    best_true = int(np.argmax(probs))
    counts = np.zeros(k)
    means = np.zeros(k)
    for a in range(k):
        means[a] = float(rng.random() < probs[a])
        counts[a] = 1.0
    t = k
    while t < max_pulls:
        b = {a: _ucb_bound(t, counts[a], delta) for a in range(k)}
        leader = int(np.argmax(means))
        challenger = max(
            (a for a in range(k) if a != leader),
            key=lambda a: means[a] + b[a],
        )
        if means[leader] - b[leader] >= means[challenger] + b[challenger]:
            return leader, t, bool(leader == best_true)
        for a in (leader, challenger):
            r = float(rng.random() < probs[a])
            counts[a] += 1.0
            means[a] += (r - means[a]) / counts[a]
        t += 1
    leader = int(np.argmax(means))
    return leader, t, bool(leader == best_true)


def bench_best_arm(seed: int | None = None) -> dict[str, float]:
    rng = np.random.default_rng(seed if seed is not None else 20261231)
    probs = np.array([0.72, 0.62, 0.5, 0.45, 0.4])
    delta = 0.05
    runs = 6
    se_ok = lu_ok = 0
    se_p = lu_p = 0.0
    for _ in range(runs):
        _, p1, ok1 = successive_elimination(probs, delta, rng)
        _, p2, ok2 = lucb(probs, delta, rng)
        se_ok += int(ok1)
        lu_ok += int(ok2)
        se_p += p1
        lu_p += p2
    return {
        "synthetic_se_correct_frac": se_ok / runs,
        "synthetic_lucb_correct_frac": lu_ok / runs,
        "synthetic_se_pulls": se_p / runs,
        "synthetic_lucb_pulls": lu_p / runs,
    }
