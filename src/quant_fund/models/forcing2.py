"""Cohen forcing: finite approximations and dense sets (SYNTHETIC)."""

from __future__ import annotations


def extends(p: dict[int, int], q: dict[int, int]) -> bool:
    """q extends p iff q sup p as partial functions."""
    return all(k in q and q[k] == v for k, v in p.items())


def meets_dense(cond: dict[int, int], dense_domain: int) -> bool:
    """A condition meets the dense set D_n = {p : n in dom p}."""
    return dense_domain in cond


def _bench_forcing2(seed: int = 0) -> float:
    checks = []
    p = {0: 1, 2: 0}
    q = {0: 1, 2: 0, 5: 1}
    checks.append(extends(p, q))
    checks.append(not extends(q, p))
    # incompatible conditions don't both extend
    checks.append(not extends({0: 1}, {0: 0}))
    # dense set D_5 met by q
    checks.append(meets_dense(q, 5))
    # generic filter meets every dense set
    checks.append(meets_dense(p, 0))
    return float(sum(checks) / len(checks))


def bench_forcing2(seed: int = 0) -> dict[str, float]:
    return {"synthetic_forcing2": _bench_forcing2(seed)}
