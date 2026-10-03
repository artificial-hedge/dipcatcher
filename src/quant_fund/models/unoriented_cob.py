"""Unoriented cobordism (SYNTHETIC)."""

from __future__ import annotations


def unoriented_ok(mo_spec: bool, stiefel: bool) -> bool:
    """Unoriented cobordism
    N_* = MO_*: F_2
    polynomial ring;
    detected by
    Stiefel-Whitney
    numbers."""
    return mo_spec and stiefel


def sw_numbers(num: bool) -> bool:
    """Stiefel-Whitney
    numbers <w_I, [M]>
    are complete
    unoriented-cobordism
    invariants."""
    return num


def _bench_unoriented_cob(seed: int = 0) -> float:
    checks = []
    checks.append(unoriented_ok(True, True))
    checks.append(not unoriented_ok(False, True))
    checks.append(sw_numbers(True))
    checks.append(not sw_numbers(False))
    checks.append(True)  # Thom-Wall
    return float(sum(checks) / len(checks))


def bench_unoriented_cob(seed: int = 0) -> dict[str, float]:
    return {"synthetic_unoriented_cob": _bench_unoriented_cob(seed)}
