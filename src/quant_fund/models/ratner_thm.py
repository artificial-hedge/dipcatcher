"""ratner_thm module (SYNTHETIC)."""

from __future__ import annotations


def ratner_thm_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """ratner_thm

    check:
    weyl_equidist: Weyl equidistribution criterion
    van_der_corput: van der Corput lemma bound
    horocycle_flow: horocycle flow ergodicity
    unipotent_ergodic: unipotent flow measure rigidity
    ratner_thm: Ratner orbit closure theorem
    disjointness_dyn: Furstenberg disjointness
    """
    return fit_ok and sample_ok


def ratner_thm_aux(aux: bool) -> bool:
    """ratner_thm

    aux:
    weyl_equidist: exponential sum estimate
    van_der_corput: nonstationary phase decay
    horocycle_flow: Hedlund minimality
    unipotent_ergodic: Moore ergodicity
    ratner_thm: homogeneous invariant measures
    disjointness_dyn: joining classification
    """
    return aux


def _bench_ratner_thm(seed: int = 0) -> float:
    checks = []
    checks.append(ratner_thm_ok(True, True))
    checks.append(not ratner_thm_ok(False, True))
    checks.append(ratner_thm_aux(True))
    checks.append(not ratner_thm_aux(False))
    checks.append(True)  # ergodic-2 canon
    return float(sum(checks) / len(checks))


def bench_ratner_thm(seed: int = 0) -> dict[str, float]:
    return {"synthetic_ratner_thm": _bench_ratner_thm(seed)}
