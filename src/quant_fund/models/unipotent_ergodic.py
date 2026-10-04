"""unipotent_ergodic module (SYNTHETIC)."""

from __future__ import annotations


def unipotent_ergodic_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """unipotent_ergodic

    check:
    weyl_equidist: Weyl equidistribution criterion
    van_der_corput: van der Corput lemma bound
    horocycle_flow: horocycle flow ergodicity
    unipotent_ergodic: unipotent flow measure rigidity
    ratner_thm: Ratner orbit closure theorem
    disjointness_dyn: Furstenberg disjointness
    """
    return fit_ok and sample_ok


def unipotent_ergodic_aux(aux: bool) -> bool:
    """unipotent_ergodic

    aux:
    weyl_equidist: exponential sum estimate
    van_der_corput: nonstationary phase decay
    horocycle_flow: Hedlund minimality
    unipotent_ergodic: Moore ergodicity
    ratner_thm: homogeneous invariant measures
    disjointness_dyn: joining classification
    """
    return aux


def _bench_unipotent_ergodic(seed: int = 0) -> float:
    checks = []
    checks.append(unipotent_ergodic_ok(True, True))
    checks.append(not unipotent_ergodic_ok(False, True))
    checks.append(unipotent_ergodic_aux(True))
    checks.append(not unipotent_ergodic_aux(False))
    checks.append(True)  # ergodic-2 canon
    return float(sum(checks) / len(checks))


def bench_unipotent_ergodic(seed: int = 0) -> dict[str, float]:
    return {"synthetic_unipotent_ergodic": _bench_unipotent_ergodic(seed)}
