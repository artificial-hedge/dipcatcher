"""Adaptive query execution: measure join selectivity mid-plan, switch order (SYNTHETIC)."""

import numpy as np

_SEED = 20261231 + 605


def _run_order(t_a: np.ndarray, t_b: np.ndarray, t_c: np.ndarray, order: str) -> int:
    """True cost proxy on full cardinalities: intermediate product."""
    j_ab = min(len(t_a), len(t_b))
    j_bc = min(len(t_b), len(t_c))
    if order == "ab_c":
        return j_ab * len(t_c)
    return j_bc * len(t_a)


def _sampled_join_size(
    t1: np.ndarray, t2: np.ndarray, rng: np.random.RandomState, frac: float
) -> float:
    """Mid-plan probe: estimate min-cardinality join from a frac sample."""
    n1 = max(1, int(round(len(t1) * frac)))
    n2 = max(1, int(round(len(t2) * frac)))
    s1 = rng.choice(len(t1), n1, replace=False)
    s2 = rng.choice(len(t2), n2, replace=False)
    return float(min(len(np.intersect1d(t1[s1], t2[s2])), min(n1, n2)) / frac)


def _adaptive_choice(
    t_a: np.ndarray,
    t_b: np.ndarray,
    t_c: np.ndarray,
    rng: np.random.RandomState,
    frac: float = 0.2,
) -> str:
    """Pick join order from sampled cardinality estimates (the adaptive step)."""
    est_ab = _sampled_join_size(t_a, t_b, rng, frac) * len(t_c)
    est_bc = _sampled_join_size(t_b, t_c, rng, frac) * len(t_a)
    return "ab_c" if est_ab <= est_bc else "bc_a"


def bench_adaptive_qp(seed: int = _SEED, frac: float = 0.35) -> dict[str, float]:
    rng = np.random.RandomState(seed)
    match = beats_worst = 0
    regret = 0.0
    trials = 40
    for _ in range(trials):
        a = np.arange(rng.randint(10, 500))
        b = np.arange(rng.randint(10, 500))
        c = np.arange(rng.randint(10, 500))
        chosen = _adaptive_choice(a, b, c, rng, frac)
        costs = {o: _run_order(a, b, c, o) for o in ("ab_c", "bc_a")}
        best = min(costs.values())
        match += int(costs[chosen] == best)
        beats_worst += int(costs[chosen] < max(costs.values()))
        regret += float(costs[chosen] - best)
    return {
        "synthetic_adaptive_oracle_match": match / trials,
        "synthetic_adaptive_beats_worst": beats_worst / trials,
        "synthetic_adaptive_regret": regret / trials,
        "synthetic_probe_frac": frac,
    }
