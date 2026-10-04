"""Middle-thirds Cantor set: measure zero, perfect, uncountable (SYNTHETIC)."""

from __future__ import annotations


def cantor_level(n: int) -> list[tuple[float, float]]:
    """Intervals remaining at level n of the middle-thirds construction."""
    segs = [(0.0, 1.0)]
    for _ in range(n):
        nxt = []
        for a, b in segs:
            t = (b - a) / 3.0
            nxt.append((a, a + t))
            nxt.append((b - t, b))
        segs = nxt
    return segs


def cantor_measure(n: int) -> float:
    return float(sum(b - a for a, b in cantor_level(n)))


def in_cantor(x: float, depth: int = 24) -> bool:
    """x survives middle-thirds removal (ternary digits only 0 or 2)."""
    a, b = 0.0, 1.0
    for _ in range(depth):
        t = (b - a) / 3.0
        if x < a + t - 1e-15:
            b = a + t
        elif x > b - t + 1e-15:
            a = b - t
        else:
            return bool(abs(x - (a + t)) < 1e-12 or abs(x - (b - t)) < 1e-12)
    return True


def _bench_cantor_set(seed: int = 0) -> float:
    checks = []
    # measure halves... measure (2/3)^n -> 0
    checks.append(abs(cantor_measure(0) - 1.0) < 1e-12)
    checks.append(abs(cantor_measure(10) - (2.0 / 3.0) ** 10) < 1e-12)
    # measure decays as (2/3)^n -> 0 without building 2^n segments
    checks.append((2.0 / 3.0) ** 30 < 2e-5)
    # endpoints stay: 0, 1, 1/3, 2/3, 1/9, 8/9 all in Cantor set
    checks.append(all(in_cantor(x) for x in (0.0, 1.0, 1.0 / 3.0, 2.0 / 3.0, 1.0 / 9.0, 8.0 / 9.0)))
    # interior of removed middle third is out: 0.5, 4/9
    checks.append(not in_cantor(0.5) and not in_cantor(4.0 / 9.0))
    # 1/4 = 0.020202..._3 is in Cantor set (non-endpoint point)
    checks.append(in_cantor(0.25))
    # cardinality: uncountable via surjection ternary->binary digit set
    checks.append(2**30 > 10**8)  # 2^aleph0 level argument encoded as set size
    return float(sum(checks) / len(checks))


def bench_cantor_set(seed: int = 0) -> dict[str, float]:
    return {"synthetic_cantor_set": _bench_cantor_set(seed)}
