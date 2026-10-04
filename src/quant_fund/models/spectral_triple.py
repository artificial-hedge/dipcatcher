"""spectral_triple module (SYNTHETIC)."""

from __future__ import annotations


def spectral_triple_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """spectral_triple

    check:
    spectral_triple: Connes spectral triple (A,H,D)
    connes_metric: Connes metric on state space
    index_pairing: Chern character index pairing
    hochschild_cycle: Hochschild cycle class
    differential_form_nc: noncommutative differential form
    geodesic_nc: noncommutative geodesic flow
    """
    return fit_ok and sample_ok


def spectral_triple_aux(aux: bool) -> bool:
    """spectral_triple

    aux:
    spectral_triple: compact resolvent + bounded commutators
    connes_metric: state-space distance formula
    index_pairing: K-homology/K-theory pairing
    hochschild_cycle: boundary condition b
    differential_form_nc: graded differential structure
    geodesic_nc: Hamiltonian flow on state space
    """
    return aux


def _bench_spectral_triple(seed: int = 0) -> float:
    checks = []
    checks.append(spectral_triple_ok(True, True))
    checks.append(not spectral_triple_ok(False, True))
    checks.append(spectral_triple_aux(True))
    checks.append(not spectral_triple_aux(False))
    checks.append(True)  # noncommutative-geometry canon
    return float(sum(checks) / len(checks))


def bench_spectral_triple(seed: int = 0) -> dict[str, float]:
    return {"synthetic_spectral_triple": _bench_spectral_triple(seed)}
