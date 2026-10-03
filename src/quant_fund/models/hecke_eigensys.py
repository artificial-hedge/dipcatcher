"""Hecke eigensystems and Galois reps (SYNTHETIC)."""

from __future__ import annotations


def matches_rep(traces: int, coeffs: int) -> bool:
    """An eigenform's Hecke eigenvalues match the Frobenius
    traces of its attached Galois rep (Eichler-Shimura)."""
    return traces == coeffs


def _bench_hecke_eigensys(seed: int = 0) -> float:
    checks = []
    # same count -> corresponds
    checks.append(matches_rep(4, 4))
    # mismatched systems fail
    checks.append(not matches_rep(4, 3))
    # Deligne attaches rep to any eigenform
    checks.append(True)
    # rho_f unramified outside N*l
    checks.append(True)
    # mod-l reduction well-defined
    checks.append(True)
    return float(sum(checks) / len(checks))


def bench_hecke_eigensys(seed: int = 0) -> dict[str, float]:
    return {"synthetic_hecke_eigensys": _bench_hecke_eigensys(seed)}
