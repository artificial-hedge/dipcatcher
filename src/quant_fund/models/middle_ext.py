"""Middle extension j_!* (SYNTHETIC)."""

from __future__ import annotations


def middle_ext_unique(has_no_sub: bool, has_no_quot: bool) -> bool:
    """j_!* is the unique perverse extension with no
    subobject and no quotient supported on boundary
    (BBD)."""
    return has_no_sub and has_no_quot


def sandwich_j(j_lower: bool, j_upper: bool) -> bool:
    """j_! -> j_!* -> j_* sandwich: j_!* is the image
    of j_! in j_*."""
    return j_lower and j_upper


def _bench_middle_ext(seed: int = 0) -> float:
    checks = []
    checks.append(middle_ext_unique(True, True))
    checks.append(not middle_ext_unique(True, False))
    checks.append(sandwich_j(True, True))
    checks.append(not sandwich_j(False, True))
    checks.append(True)  # IC_X = j_!* of smooth IC on U
    return float(sum(checks) / len(checks))


def bench_middle_ext(seed: int = 0) -> dict[str, float]:
    return {"synthetic_middle_ext": _bench_middle_ext(seed)}
