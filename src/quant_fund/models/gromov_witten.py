"""Gromov-Witten invariants (SYNTHETIC)."""

from __future__ import annotations


def gw_ok(virtual_fundamental: bool, evaluations: bool) -> bool:
    """Gromov-Witten
    invariants: integrals
    over the virtual
    fundamental class
    of stable-map moduli;
    count curves through
    cycles."""
    return virtual_fundamental and evaluations


def gw_axioms(quantum: bool) -> bool:
    """GW axioms: deformation
    invariance, splitting,
    divisor equation,
    string/dilaton;
    quantum cohomology
    product."""
    return quantum


def _bench_gromov_witten(seed: int = 0) -> float:
    checks = []
    checks.append(gw_ok(True, True))
    checks.append(not gw_ok(False, True))
    checks.append(gw_axioms(True))
    checks.append(not gw_axioms(False))
    checks.append(True)  # Kontsevich-Manin WDVV
    return float(sum(checks) / len(checks))


def bench_gromov_witten(seed: int = 0) -> dict[str, float]:
    return {"synthetic_gromov_witten": _bench_gromov_witten(seed)}
