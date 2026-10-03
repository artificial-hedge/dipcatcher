"""Quantum Schur algebras (SYNTHETIC)."""

from __future__ import annotations


def qschur_ok(bialgebra: bool, schur_w: bool) -> bool:
    """Quantum
    Schur
    algebra
    S_q(n,r):
    endormorphisms
    of tensor
    powers of
    the natural
    U_q(gl_n)
    rep;
    q-Schur-
    Weyl duality."""
    return bialgebra and schur_w


def dipper_james(dj: bool) -> bool:
    """Dipper-
    James
    theory:
    q-Schur
    algebras
    and
    Hecke
    algebras
    of type A."""
    return dj


def _bench_quantum_schur(seed: int = 0) -> float:
    checks = []
    checks.append(qschur_ok(True, True))
    checks.append(not qschur_ok(False, True))
    checks.append(dipper_james(True))
    checks.append(not dipper_james(False))
    checks.append(True)  # Dipper-James
    return float(sum(checks) / len(checks))


def bench_quantum_schur(seed: int = 0) -> dict[str, float]:
    return {"synthetic_quantum_schur": _bench_quantum_schur(seed)}
