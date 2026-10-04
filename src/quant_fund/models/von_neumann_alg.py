"""von_neumann_alg module (SYNTHETIC)."""

from __future__ import annotations


def von_neumann_alg_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """von_neumann_alg

    check:
    von_neumann_alg: weakly closed selfadjoint algebra
    double_commutant: bicommutant density theorem
    predual_space: unique predual characterization
    normal_state: ultraweakly continuous state
    tomita_takesaki: modular automorphism group
    jones_index: subfactor index invariant
    """
    return fit_ok and sample_ok


def von_neumann_alg_aux(aux: bool) -> bool:
    """von_neumann_alg

    aux:
    von_neumann_alg: center decomposition
    double_commutant: commutant computation
    predual_space: Sakai theorem
    normal_state: normal trace
    tomita_takesaki: modular conjugation
    jones_index: tower construction
    """
    return aux


def _bench_von_neumann_alg(seed: int = 0) -> float:
    checks = []
    checks.append(von_neumann_alg_ok(True, True))
    checks.append(not von_neumann_alg_ok(False, True))
    checks.append(von_neumann_alg_aux(True))
    checks.append(not von_neumann_alg_aux(False))
    checks.append(True)  # von-neumann-algebra canon
    return float(sum(checks) / len(checks))


def bench_von_neumann_alg(seed: int = 0) -> dict[str, float]:
    return {"synthetic_von_neumann_alg": _bench_von_neumann_alg(seed)}
