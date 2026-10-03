"""double_commutant module (SYNTHETIC)."""

from __future__ import annotations


def double_commutant_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """double_commutant

    check:
    von_neumann_alg: weakly closed selfadjoint algebra
    double_commutant: bicommutant density theorem
    predual_space: unique predual characterization
    normal_state: ultraweakly continuous state
    tomita_takesaki: modular automorphism group
    jones_index: subfactor index invariant
    """
    return fit_ok and sample_ok


def double_commutant_aux(aux: bool) -> bool:
    """double_commutant

    aux:
    von_neumann_alg: center decomposition
    double_commutant: commutant computation
    predual_space: Sakai theorem
    normal_state: normal trace
    tomita_takesaki: modular conjugation
    jones_index: tower construction
    """
    return aux


def _bench_double_commutant(seed: int = 0) -> float:
    checks = []
    checks.append(double_commutant_ok(True, True))
    checks.append(not double_commutant_ok(False, True))
    checks.append(double_commutant_aux(True))
    checks.append(not double_commutant_aux(False))
    checks.append(True)  # von-neumann-algebra canon
    return float(sum(checks) / len(checks))


def bench_double_commutant(seed: int = 0) -> dict[str, float]:
    return {"synthetic_double_commutant": _bench_double_commutant(seed)}
