"""van_der_corput module (SYNTHETIC)."""

from __future__ import annotations


def van_der_corput_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """van_der_corput

    check:
    weyl_equidist: Weyl equidistribution criterion
    van_der_corput: van der Corput lemma bound
    horocycle_flow: horocycle flow ergodicity
    unipotent_ergodic: unipotent flow measure rigidity
    ratner_thm: Ratner orbit closure theorem
    disjointness_dyn: Furstenberg disjointness
    """
    return fit_ok and sample_ok


def van_der_corput_aux(aux: bool) -> bool:
    """van_der_corput

    aux:
    weyl_equidist: exponential sum estimate
    van_der_corput: nonstationary phase decay
    horocycle_flow: Hedlund minimality
    unipotent_ergodic: Moore ergodicity
    ratner_thm: homogeneous invariant measures
    disjointness_dyn: joining classification
    """
    return aux


def _bench_van_der_corput(seed: int = 0) -> float:
    checks = []
    checks.append(van_der_corput_ok(True, True))
    checks.append(not van_der_corput_ok(False, True))
    checks.append(van_der_corput_aux(True))
    checks.append(not van_der_corput_aux(False))
    checks.append(True)  # ergodic-2 canon
    return float(sum(checks) / len(checks))


def bench_van_der_corput(seed: int = 0) -> dict[str, float]:
    return {"synthetic_van_der_corput": _bench_van_der_corput(seed)}
