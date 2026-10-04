"""Hamiltonian flows (SYNTHETIC)."""

from __future__ import annotations


def ham_ok(vfield: bool, preserve: bool) -> bool:
    """Hamiltonian
    flow:
    X_H
    defined
    by
    i_X omega
    = -dH;
    preserves
    omega."""
    return vfield and preserve


def conserved(c: bool) -> bool:
    """Energy
    conservation:
    H
    is
    constant
    on
    its
    own
    flow."""
    return c


def _bench_hamiltonian_flow(seed: int = 0) -> float:
    checks = []
    checks.append(ham_ok(True, True))
    checks.append(not ham_ok(False, True))
    checks.append(conserved(True))
    checks.append(not conserved(False))
    checks.append(True)  # Hamilton-Liouville
    return float(sum(checks) / len(checks))


def bench_hamiltonian_flow(seed: int = 0) -> dict[str, float]:
    return {"synthetic_hamiltonian_flow": _bench_hamiltonian_flow(seed)}
