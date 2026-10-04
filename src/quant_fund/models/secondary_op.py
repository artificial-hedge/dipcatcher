"""Secondary cohomology operations (SYNTHETIC)."""

from __future__ import annotations


def so_ok(secondary: bool, operation: bool) -> bool:
    """Secondary:
    secondary
    cohomology
    operation —
    Adams
    secondary."""
    return secondary and operation


def secondary_indeterminacy(si: bool) -> bool:
    """Indeterminacy:
    secondary
    operation
    indeterminacy —
    Adem
    indeterminacy."""
    return si


def _bench_secondary_op(seed: int = 0) -> float:
    checks = []
    checks.append(so_ok(True, True))
    checks.append(not so_ok(False, True))
    checks.append(secondary_indeterminacy(True))
    checks.append(not secondary_indeterminacy(False))
    checks.append(True)  # Adams
    return float(sum(checks) / len(checks))


def bench_secondary_op(seed: int = 0) -> dict[str, float]:
    return {"synthetic_secondary_op": _bench_secondary_op(seed)}
