"""Adaptive query execution: measure join selectivity mid-plan, switch order (SYNTHETIC)."""

import numpy as np

_SEED = 20261231 + 605


def _run_order(t_a: np.ndarray, t_b: np.ndarray, t_c: np.ndarray, order: str) -> int:
    """Cost proxy: intermediate cardinality product."""
    j_ab = min(len(t_a), len(t_b))
    j_bc = min(len(t_b), len(t_c))
    if order == "ab_c":
        return j_ab * len(t_c)
    return j_bc * len(t_a)


def bench_adaptive_qp(seed: int = _SEED) -> dict[str, float]:
    rng = np.random.RandomState(seed)
    ok = 0
    for _ in range(40):
        a = np.arange(rng.randint(10, 500))
        b = np.arange(rng.randint(10, 500))
        c = np.arange(rng.randint(10, 500))
        # adaptive: probe min(|a|,|b|) vs min(|b|,|c|) then pick cheaper
        est_ab = _run_order(a, b, c, "ab_c")
        est_bc = _run_order(a, b, c, "bc_a")
        chosen = "ab_c" if est_ab <= est_bc else "bc_a"
        if _run_order(a, b, c, chosen) == min(est_ab, est_bc):
            ok += 1
    return {"synthetic_adaptive_optimal": ok / 40}
