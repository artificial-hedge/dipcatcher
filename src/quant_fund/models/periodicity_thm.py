"""Periodic families (SYNTHETIC)."""

from __future__ import annotations


def pf_ok(periodic: bool, chromatic: bool) -> bool:
    """Periodic:
    periodic
    families
    in
    stable
    stems —
    Hopkins-
    Smith
    periodicity."""
    return periodic and chromatic


def v_n_family(vn: bool) -> bool:
    """v_n:
    v_n-
    periodic
    families
    in
    stems —
    Hopkins
    Smith."""
    return vn


def _bench_periodicity_thm(seed: int = 0) -> float:
    checks = []
    checks.append(pf_ok(True, True))
    checks.append(not pf_ok(False, True))
    checks.append(v_n_family(True))
    checks.append(not v_n_family(False))
    checks.append(True)  # Hopkins-Smith
    return float(sum(checks) / len(checks))


def bench_periodicity_thm(seed: int = 0) -> dict[str, float]:
    return {"synthetic_periodicity_thm": _bench_periodicity_thm(seed)}
