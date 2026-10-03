"""Covering-space lifting obstruction (SYNTHETIC)."""

from __future__ import annotations


def lifts(f_index: int, p_index: int) -> bool:
    """f: X -> B lifts through p: E -> B iff f_*(pi1 X) subset p_*(pi1 E).
    With pi1 = Z, images are subgroups f_index*Z and p_index*Z;
    inclusion iff p_index divides f_index."""
    if p_index == 0:
        return f_index == 0
    return f_index % p_index == 0


def _bench_obstruction_toy(seed: int = 0) -> float:
    checks = []
    # identity map (f_*=Z) lifts only to trivial cover
    checks.append(lifts(1, 1))
    # degree-2 map into S^1 lifts through the 2-fold cover p_*=2Z
    checks.append(lifts(4, 2))
    # degree-2 does NOT lift through 3-fold cover
    checks.append(not lifts(2, 3))
    # null-homotopic map (f_*=0) lifts to every cover
    checks.append(lifts(0, 5))
    # universal cover (p_*=0) only for trivial image
    checks.append(not lifts(2, 0))
    checks.append(lifts(0, 0))
    return float(sum(checks) / len(checks))


def bench_obstruction_toy(seed: int = 0) -> dict[str, float]:
    return {"synthetic_obstruction_toy": _bench_obstruction_toy(seed)}
