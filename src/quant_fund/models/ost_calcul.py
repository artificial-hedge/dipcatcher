"""Optional stopping on asymmetric random walk (SYNTHETIC)."""

from __future__ import annotations


def ruin_prob(p: float, i: int, n: int) -> float:
    """Gambler's ruin: P(hit 0 before n | start i, up-prob p)."""
    if p == 0.5:
        return 1.0 - i / n
    r = (1 - p) / p
    return (r**i - r**n) / (1 - r**n)


def exp_duration(p: float, i: int, n: int) -> float:
    """E[tau] for absorbing walk on {0..n}."""
    # solve linear system E_i = 1 + p E_{i+1} + (1-p) E_{i-1}
    # via tridiagonal solve on i = 1..n-1
    import numpy as np

    m = n - 1
    if m <= 0:
        return 0.0
    a = np.zeros((m, m))
    b = np.ones(m)
    for k in range(m):
        j = k + 1
        a[k, k] = 1.0
        if j + 1 < n:
            a[k, k + 1] = -p
        if j - 1 > 0:
            a[k, k - 1] = -(1 - p)
    sol = np.linalg.solve(a, b)
    return float(sol[i - 1])


def _bench_ost_calcul(seed: int = 0) -> float:
    checks = []
    # symmetric ruin prob = 1 - i/n
    checks.append(abs(ruin_prob(0.5, 3, 10) - 0.7) < 1e-12)
    # asymmetric: p > 1/2 reduces ruin probability
    checks.append(ruin_prob(0.6, 5, 10) < ruin_prob(0.5, 5, 10))
    # boundary conditions
    checks.append(ruin_prob(0.4, 0, 10) == 1.0)
    checks.append(ruin_prob(0.4, 10, 10) == 0.0)
    # symmetric duration i(n-i)
    checks.append(abs(exp_duration(0.5, 3, 10) - 21.0) < 1e-9)
    return float(sum(checks) / len(checks))


def bench_ost_calcul(seed: int = 0) -> dict[str, float]:
    return {"synthetic_ost_calcul": _bench_ost_calcul(seed)}
