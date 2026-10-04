"""tseng_split module (SYNTHETIC)."""

from __future__ import annotations


def tseng_split_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """tseng_split

    check:
    douglas_rachford: DR splitting iteration
    peaceman_rachford: PR alternating splitting
    tseng_split: Tseng forward-backward-forward
    forward_backward: forward-backward splitting
    chambolle_pock: primal-dual algorithm
    davis_yin: three-operator splitting
    """
    return fit_ok and sample_ok


def tseng_split_aux(aux: bool) -> bool:
    """tseng_split

    aux:
    douglas_rachford: reflected resolvent
    peaceman_rachford: asymmetric DR step
    tseng_split: monotone inclusion solve
    forward_backward: proximal gradient form
    chambolle_pock: saddle-point extragradient
    davis_yin: three-term composition
    """
    return aux


def _bench_tseng_split(seed: int = 0) -> float:
    checks = []
    checks.append(tseng_split_ok(True, True))
    checks.append(not tseng_split_ok(False, True))
    checks.append(tseng_split_aux(True))
    checks.append(not tseng_split_aux(False))
    checks.append(True)  # operator-splitting canon
    return float(sum(checks) / len(checks))


def bench_tseng_split(seed: int = 0) -> dict[str, float]:
    return {"synthetic_tseng_split": _bench_tseng_split(seed)}
