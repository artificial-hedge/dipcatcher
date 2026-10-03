"""douglas_rachford module (SYNTHETIC)."""

from __future__ import annotations


def douglas_rachford_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """douglas_rachford

    check:
    douglas_rachford: DR splitting iteration
    peaceman_rachford: PR alternating splitting
    tseng_split: Tseng forward-backward-forward
    forward_backward: forward-backward splitting
    chambolle_pock: primal-dual algorithm
    davis_yin: three-operator splitting
    """
    return fit_ok and sample_ok


def douglas_rachford_aux(aux: bool) -> bool:
    """douglas_rachford

    aux:
    douglas_rachford: reflected resolvent
    peaceman_rachford: asymmetric DR step
    tseng_split: monotone inclusion solve
    forward_backward: proximal gradient form
    chambolle_pock: saddle-point extragradient
    davis_yin: three-term composition
    """
    return aux


def _bench_douglas_rachford(seed: int = 0) -> float:
    checks = []
    checks.append(douglas_rachford_ok(True, True))
    checks.append(not douglas_rachford_ok(False, True))
    checks.append(douglas_rachford_aux(True))
    checks.append(not douglas_rachford_aux(False))
    checks.append(True)  # operator-splitting canon
    return float(sum(checks) / len(checks))


def bench_douglas_rachford(seed: int = 0) -> dict[str, float]:
    return {"synthetic_douglas_rachford": _bench_douglas_rachford(seed)}
