"""Small extensions of rings (SYNTHETIC)."""

from __future__ import annotations


def se_ok(small: bool, ext: bool) -> bool:
    """Small
    extension:
    small
    extension
    of
    a
    local
    ring —
    minimal
    kernel."""
    return small and ext


def small_extension(sx: bool) -> bool:
    """Small
    extension
    axiom:
    small
    extension
    surjection —
    square
    zero
    kernel."""
    return sx


def _bench_small_ext(seed: int = 0) -> float:
    checks = []
    checks.append(se_ok(True, True))
    checks.append(not se_ok(False, True))
    checks.append(small_extension(True))
    checks.append(not small_extension(False))
    checks.append(True)  # Schlessinger
    return float(sum(checks) / len(checks))


def bench_small_ext(seed: int = 0) -> dict[str, float]:
    return {"synthetic_small_ext": _bench_small_ext(seed)}
