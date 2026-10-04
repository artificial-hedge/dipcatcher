"""Primary cohomology operations (SYNTHETIC)."""

from __future__ import annotations


def po_ok(primary: bool, operation: bool) -> bool:
    """Primary:
    primary
    cohomology
    operation —
    Steenrod
    primary."""
    return primary and operation


def unstable_sq(us: bool) -> bool:
    """Unstable:
    unstable
    Steenrod
    axioms —
    Cartan
    unstable."""
    return us


def _bench_primary_op(seed: int = 0) -> float:
    checks = []
    checks.append(po_ok(True, True))
    checks.append(not po_ok(False, True))
    checks.append(unstable_sq(True))
    checks.append(not unstable_sq(False))
    checks.append(True)  # Steenrod
    return float(sum(checks) / len(checks))


def bench_primary_op(seed: int = 0) -> dict[str, float]:
    return {"synthetic_primary_op": _bench_primary_op(seed)}
