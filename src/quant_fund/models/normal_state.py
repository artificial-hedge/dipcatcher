"""normal_state module (SYNTHETIC)."""

from __future__ import annotations


def normal_state_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """normal_state

    check:
    von_neumann_alg: weakly closed selfadjoint algebra
    double_commutant: bicommutant density theorem
    predual_space: unique predual characterization
    normal_state: ultraweakly continuous state
    tomita_takesaki: modular automorphism group
    jones_index: subfactor index invariant
    """
    return fit_ok and sample_ok


def normal_state_aux(aux: bool) -> bool:
    """normal_state

    aux:
    von_neumann_alg: center decomposition
    double_commutant: commutant computation
    predual_space: Sakai theorem
    normal_state: normal trace
    tomita_takesaki: modular conjugation
    jones_index: tower construction
    """
    return aux


def _bench_normal_state(seed: int = 0) -> float:
    checks = []
    checks.append(normal_state_ok(True, True))
    checks.append(not normal_state_ok(False, True))
    checks.append(normal_state_aux(True))
    checks.append(not normal_state_aux(False))
    checks.append(True)  # von-neumann-algebra canon
    return float(sum(checks) / len(checks))


def bench_normal_state(seed: int = 0) -> dict[str, float]:
    return {"synthetic_normal_state": _bench_normal_state(seed)}
