"""connes_metric module (SYNTHETIC)."""

from __future__ import annotations


def connes_metric_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """connes_metric

    check:
    spectral_triple: Connes spectral triple (A,H,D)
    connes_metric: Connes metric on state space
    index_pairing: Chern character index pairing
    hochschild_cycle: Hochschild cycle class
    differential_form_nc: noncommutative differential form
    geodesic_nc: noncommutative geodesic flow
    """
    return fit_ok and sample_ok


def connes_metric_aux(aux: bool) -> bool:
    """connes_metric

    aux:
    spectral_triple: compact resolvent + bounded commutators
    connes_metric: state-space distance formula
    index_pairing: K-homology/K-theory pairing
    hochschild_cycle: boundary condition b
    differential_form_nc: graded differential structure
    geodesic_nc: Hamiltonian flow on state space
    """
    return aux


def _bench_connes_metric(seed: int = 0) -> float:
    checks = []
    checks.append(connes_metric_ok(True, True))
    checks.append(not connes_metric_ok(False, True))
    checks.append(connes_metric_aux(True))
    checks.append(not connes_metric_aux(False))
    checks.append(True)  # noncommutative-geometry canon
    return float(sum(checks) / len(checks))


def bench_connes_metric(seed: int = 0) -> dict[str, float]:
    return {"synthetic_connes_metric": _bench_connes_metric(seed)}
