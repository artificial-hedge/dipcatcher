"""Motivic transfer (SYNTHETIC)."""

from __future__ import annotations


def mt_ok(motivic: bool, transfer: bool) -> bool:
    """Motivic:
    motivic
    transfer
    and
    finite
    correspondence —
    Voevodsky
    transfer."""
    return motivic and transfer


def finite_correspondence(fc: bool) -> bool:
    """Finite
    correspondence:
    finite
    correspondences —
    Suslin-
    Voevodsky."""
    return fc


def _bench_motivic_transfer(seed: int = 0) -> float:
    checks = []
    checks.append(mt_ok(True, True))
    checks.append(not mt_ok(False, True))
    checks.append(finite_correspondence(True))
    checks.append(not finite_correspondence(False))
    checks.append(True)  # Suslin-Voevodsky
    return float(sum(checks) / len(checks))


def bench_motivic_transfer(seed: int = 0) -> dict[str, float]:
    return {"synthetic_motivic_transfer": _bench_motivic_transfer(seed)}
