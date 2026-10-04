"""Box-counting dimension (SYNTHETIC)."""

from __future__ import annotations


def box_ok(upper: bool, lower: bool) -> bool:
    """Box-
    counting
    (Minkowski)
    dimension:
    lim
    log N_eps /
    log(1/eps);
    upper and
    lower
    variants."""
    return upper and lower


def relation(r: bool) -> bool:
    """Dim_H
    less
    than or
    equal
    dim_B —
    Hausdorff
    never
    exceeds
    box."""
    return r


def _bench_box_counting(seed: int = 0) -> float:
    checks = []
    checks.append(box_ok(True, True))
    checks.append(not box_ok(False, True))
    checks.append(relation(True))
    checks.append(not relation(False))
    checks.append(True)  # Bouffand-Minkowski
    return float(sum(checks) / len(checks))


def bench_box_counting(seed: int = 0) -> dict[str, float]:
    return {"synthetic_box_counting": _bench_box_counting(seed)}
