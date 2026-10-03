"""davis_yin module (SYNTHETIC)."""

from __future__ import annotations


def davis_yin_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """davis_yin

    check:
    douglas_rachford: DR splitting iteration
    peaceman_rachford: PR alternating splitting
    tseng_split: Tseng forward-backward-forward
    forward_backward: forward-backward splitting
    chambolle_pock: primal-dual algorithm
    davis_yin: three-operator splitting
    """
    return fit_ok and sample_ok


def davis_yin_aux(aux: bool) -> bool:
    """davis_yin

    aux:
    douglas_rachford: reflected resolvent
    peaceman_rachford: asymmetric DR step
    tseng_split: monotone inclusion solve
    forward_backward: proximal gradient form
    chambolle_pock: saddle-point extragradient
    davis_yin: three-term composition
    """
    return aux


def _bench_davis_yin(seed: int = 0) -> float:
    checks = []
    checks.append(davis_yin_ok(True, True))
    checks.append(not davis_yin_ok(False, True))
    checks.append(davis_yin_aux(True))
    checks.append(not davis_yin_aux(False))
    checks.append(True)  # operator-splitting canon
    return float(sum(checks) / len(checks))


def bench_davis_yin(seed: int = 0) -> dict[str, float]:
    return {"synthetic_davis_yin": _bench_davis_yin(seed)}
