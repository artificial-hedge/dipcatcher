"""Weak convergence / Portmanteau (wave 288).

Empirical measure mu_n -> mu weakly iff E_mu_n[f] -> E_mu[f] for all
bounded Lipschitz f — verified for f in a Lipschitz test family as the
empirical CDF converges.
"""

import numpy as np

_SEED = 20261231 + 815


def _emp_expect(samples: np.ndarray, f) -> float:
    return float(np.mean(f(samples)))


def bench_weak_conv(seed: int = _SEED) -> dict[str, float]:
    rng = np.random.RandomState(seed)
    tests = [lambda x: np.sin(x), lambda x: np.clip(x, -1, 1), lambda x: np.cos(2 * x)]
    wants = [0.0, 0.0, np.exp(-2.0)]  # N(0,1): E sin=0, E clip~0, E cos(2X)=e^{-2}
    # cos(2x): characteristic function at 2 -> e^{-2}
    ok = 0
    for f, w in zip(tests, wants, strict=True):
        big = _emp_expect(rng.randn(80000), f)
        ok += int(abs(big - w) < 0.02)
    return {"synthetic_weak_conv": float(ok == 3)}
