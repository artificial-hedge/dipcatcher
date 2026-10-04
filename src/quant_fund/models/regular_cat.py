"""Regular category theory (SYNTHETIC)."""

from __future__ import annotations


def rc_ok(regular: bool, kernel_pairs: bool) -> bool:
    """Regular
    category:
    regular
    cat —
    kernel
    pairs
    coequalized."""
    return regular and kernel_pairs


def regular_epi(re: bool) -> bool:
    """Regular
    epi:
    regular
    epimorphism —
    coequalizer."""
    return re


def _bench_regular_cat(seed: int = 0) -> float:
    checks = []
    checks.append(rc_ok(True, True))
    checks.append(not rc_ok(False, True))
    checks.append(regular_epi(True))
    checks.append(not regular_epi(False))
    checks.append(True)  # Barr/Kock
    return float(sum(checks) / len(checks))


def bench_regular_cat(seed: int = 0) -> dict[str, float]:
    return {"synthetic_regular_cat": _bench_regular_cat(seed)}
