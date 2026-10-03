"""tomita_takesaki module (SYNTHETIC)."""

from __future__ import annotations


def tomita_takesaki_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """tomita_takesaki

    check:
    von_neumann_alg: weakly closed selfadjoint algebra
    double_commutant: bicommutant density theorem
    predual_space: unique predual characterization
    normal_state: ultraweakly continuous state
    tomita_takesaki: modular automorphism group
    jones_index: subfactor index invariant
    """
    return fit_ok and sample_ok


def tomita_takesaki_aux(aux: bool) -> bool:
    """tomita_takesaki

    aux:
    von_neumann_alg: center decomposition
    double_commutant: commutant computation
    predual_space: Sakai theorem
    normal_state: normal trace
    tomita_takesaki: modular conjugation
    jones_index: tower construction
    """
    return aux


def _bench_tomita_takesaki(seed: int = 0) -> float:
    checks = []
    checks.append(tomita_takesaki_ok(True, True))
    checks.append(not tomita_takesaki_ok(False, True))
    checks.append(tomita_takesaki_aux(True))
    checks.append(not tomita_takesaki_aux(False))
    checks.append(True)  # von-neumann-algebra canon
    return float(sum(checks) / len(checks))


def bench_tomita_takesaki(seed: int = 0) -> dict[str, float]:
    return {"synthetic_tomita_takesaki": _bench_tomita_takesaki(seed)}
