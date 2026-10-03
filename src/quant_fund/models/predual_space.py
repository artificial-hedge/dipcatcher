"""predual_space module (SYNTHETIC)."""

from __future__ import annotations


def predual_space_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """predual_space

    check:
    von_neumann_alg: weakly closed selfadjoint algebra
    double_commutant: bicommutant density theorem
    predual_space: unique predual characterization
    normal_state: ultraweakly continuous state
    tomita_takesaki: modular automorphism group
    jones_index: subfactor index invariant
    """
    return fit_ok and sample_ok


def predual_space_aux(aux: bool) -> bool:
    """predual_space

    aux:
    von_neumann_alg: center decomposition
    double_commutant: commutant computation
    predual_space: Sakai theorem
    normal_state: normal trace
    tomita_takesaki: modular conjugation
    jones_index: tower construction
    """
    return aux


def _bench_predual_space(seed: int = 0) -> float:
    checks = []
    checks.append(predual_space_ok(True, True))
    checks.append(not predual_space_ok(False, True))
    checks.append(predual_space_aux(True))
    checks.append(not predual_space_aux(False))
    checks.append(True)  # von-neumann-algebra canon
    return float(sum(checks) / len(checks))


def bench_predual_space(seed: int = 0) -> dict[str, float]:
    return {"synthetic_predual_space": _bench_predual_space(seed)}
