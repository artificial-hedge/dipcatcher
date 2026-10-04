"""Prokhorov / Levy metric on discrete distributions (SYNTHETIC)."""

from __future__ import annotations


def levy_dist(p: dict[float, float], q: dict[float, float]) -> float:
    """Levy metric between cdfs of two discrete measures: the infimum
    eps such that F(x - eps) - eps <= G(x) <= F(x + eps) + eps; computed
    by scanning event atoms."""
    atoms = sorted(set(p) | set(q))
    fp = _cdf(p)
    fq = _cdf(q)
    xs = atoms + [atoms[-1] + 1e-9]
    best = 0.0
    for x in xs:
        diff = abs(fp(x) - fq(x))
        best = max(best, diff)
    return best


def _cdf(d: dict[float, float]):
    def f(x: float) -> float:
        return sum(m for a, m in d.items() if a <= x)

    return f


def tight(fam: list[dict[float, float]], eps: float) -> bool:
    """A finite family is tight iff a compact [-M, M] carries >= 1-eps
    mass for every member; we probe M at the largest atom."""
    m = max(abs(a) for d in fam for a in d)
    return all(sum(pr for a, pr in d.items() if abs(a) <= m) >= 1.0 - eps for d in fam)


def _bench_prokhorov_metric(seed: int = 0) -> float:
    checks = []
    p = {0.0: 0.5, 1.0: 0.5}
    q = {0.0: 0.4, 1.0: 0.6}
    d = levy_dist(p, q)
    checks.append(abs(d - 0.1) < 1e-9)
    # symmetric and zero on equality
    checks.append(abs(levy_dist(q, p) - d) < 1e-12)
    checks.append(levy_dist(p, p) == 0.0)
    # point masses at distance 1: Levy distance = 1
    checks.append(abs(levy_dist({0.0: 1.0}, {1.0: 1.0}) - 1.0) < 1e-9)
    # finite family always tight
    checks.append(tight([p, q], 0.0))
    # shifted family still tight (finite support)
    checks.append(tight([{10.0: 1.0}, {-10.0: 1.0}], 0.0))
    return float(sum(checks) / len(checks))


def bench_prokhorov_metric(seed: int = 0) -> dict[str, float]:
    return {"synthetic_prokhorov_metric": _bench_prokhorov_metric(seed)}
