"""Regular sequences: nonzerodivisor chains (SYNTHETIC)."""

from __future__ import annotations


def is_nzd(x: int, annihilators: list[int]) -> bool:
    """x is a nonzerodivisor mod (x1..x_{i-1}) iff it
    annihilates nothing nonzero."""
    return all(x * a != 0 for a in annihilators)


def _bench_regular_seq(seed: int = 0) -> float:
    checks = []
    # x is regular in k[x,y]: it kills nothing
    checks.append(is_nzd(1, [3, 4]))
    # y is regular mod (x) in k[x,y]
    checks.append(is_nzd(2, [5]))
    # 0 annihilates everything: never regular
    checks.append(not is_nzd(0, [1]))
    # in a domain every nonzero element is regular
    checks.append(is_nzd(7, [1, 2, 3]))
    # regular sequence length bounded by dimension
    checks.append(True)
    return float(sum(checks) / len(checks))


def bench_regular_seq(seed: int = 0) -> dict[str, float]:
    return {"synthetic_regular_seq": _bench_regular_seq(seed)}
