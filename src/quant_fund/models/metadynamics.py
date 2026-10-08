"""Well-tempered metadynamics on a double-well potential (SYNTHETIC).

Plain MH dynamics on V(x) + V_bias(x) where V_bias is a sum of
Gaussian hills deposited at the current position every `rate` steps.
Hills fill the starting well, forcing barrier crossings; we count the
fraction of samples in the opposite well vs an unbiased baseline.
"""

import numpy as np

SIG = 0.3
HILL_H = 0.4
HILL_W = 0.4


def _v(x: float) -> float:
    return 0.25 * x**4 - 2.0 * x**2 + 1.0


def _bias(x: float, hills: np.ndarray) -> float:
    if len(hills) == 0:
        return 0.0
    return float(HILL_H * np.sum(np.exp(-((x - hills) ** 2) / (2 * HILL_W**2))))


def _run(seed: int, n: int, meta: bool) -> tuple[float, int]:
    rng = np.random.default_rng(seed)
    x = -2.0
    hills: list[float] = []
    visits = 0
    crosses = 0
    side = -1
    for t in range(n):
        prop = x + rng.normal(0, SIG)
        hv = np.asarray(hills)
        d = (_v(prop) + _bias(prop, hv)) - (_v(x) + _bias(x, hv))
        if rng.random() < np.exp(-d):
            x = prop
        if meta and t % 100 == 0:
            hills.append(x)
        if x > 0.5:
            visits += 1
        if (x > 0) != (side > 0):
            side = 1 if x > 0 else -1
            crosses += 1
    return float(visits / n), crosses


def bench_metadynamics(seed: int = 5607) -> dict[str, float]:
    frac_meta, crosses_meta = _run(seed, 30000, True)
    frac_base, crosses_base = _run(seed + 1, 30000, False)
    return {
        "synthetic_md_visit_frac": frac_meta,
        "synthetic_md_base_frac": frac_base,
        "synthetic_md_crosses": float(crosses_meta),
        "synthetic_md_base_crosses": float(crosses_base),
        "synthetic_md_gain": float(frac_meta > frac_base),
    }
