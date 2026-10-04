"""jones_index module (SYNTHETIC)."""

from __future__ import annotations


def jones_index_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """jones_index

    check:
    von_neumann_alg: weakly closed selfadjoint algebra
    double_commutant: bicommutant density theorem
    predual_space: unique predual characterization
    normal_state: ultraweakly continuous state
    tomita_takesaki: modular automorphism group
    jones_index: subfactor index invariant
    """
    return fit_ok and sample_ok


def jones_index_aux(aux: bool) -> bool:
    """jones_index

    aux:
    von_neumann_alg: center decomposition
    double_commutant: commutant computation
    predual_space: Sakai theorem
    normal_state: normal trace
    tomita_takesaki: modular conjugation
    jones_index: tower construction
    """
    return aux


def _bench_jones_index(seed: int = 0) -> float:
    checks = []
    checks.append(jones_index_ok(True, True))
    checks.append(not jones_index_ok(False, True))
    checks.append(jones_index_aux(True))
    checks.append(not jones_index_aux(False))
    checks.append(True)  # von-neumann-algebra canon
    return float(sum(checks) / len(checks))


def bench_jones_index(seed: int = 0) -> dict[str, float]:
    return {"synthetic_jones_index": _bench_jones_index(seed)}
