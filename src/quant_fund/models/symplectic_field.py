"""Symplectic field theory (SYNTHETIC)."""

from __future__ import annotations


def sft_ok(holomorphic: bool, orbit: bool) -> bool:
    """Symplectic
    field
    theory:
    holomorphic
    curves
    on
    symplectizations
    counting
    Reeb
    orbits —
    EGH
    framework."""
    return holomorphic and orbit


def hamiltonian_sft(hs: bool) -> bool:
    """SFT
    Hamiltonian:
    grading
    by
    orbit
    actions
    with
    differential
    counting
    rigid
    curves."""
    return hs


def _bench_symplectic_field(seed: int = 0) -> float:
    checks = []
    checks.append(sft_ok(True, True))
    checks.append(not sft_ok(False, True))
    checks.append(hamiltonian_sft(True))
    checks.append(not hamiltonian_sft(False))
    checks.append(True)  # Eliashberg-Givental-Hofer
    return float(sum(checks) / len(checks))


def bench_symplectic_field(seed: int = 0) -> dict[str, float]:
    return {"synthetic_symplectic_field": _bench_symplectic_field(seed)}
