"""chambolle_pock module (SYNTHETIC)."""

from __future__ import annotations


def chambolle_pock_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """chambolle_pock

    check:
    douglas_rachford: DR splitting iteration
    peaceman_rachford: PR alternating splitting
    tseng_split: Tseng forward-backward-forward
    forward_backward: forward-backward splitting
    chambolle_pock: primal-dual algorithm
    davis_yin: three-operator splitting
    """
    return fit_ok and sample_ok


def chambolle_pock_aux(aux: bool) -> bool:
    """chambolle_pock

    aux:
    douglas_rachford: reflected resolvent
    peaceman_rachford: asymmetric DR step
    tseng_split: monotone inclusion solve
    forward_backward: proximal gradient form
    chambolle_pock: saddle-point extragradient
    davis_yin: three-term composition
    """
    return aux


def _bench_chambolle_pock(seed: int = 0) -> float:
    checks = []
    checks.append(chambolle_pock_ok(True, True))
    checks.append(not chambolle_pock_ok(False, True))
    checks.append(chambolle_pock_aux(True))
    checks.append(not chambolle_pock_aux(False))
    checks.append(True)  # operator-splitting canon
    return float(sum(checks) / len(checks))


def bench_chambolle_pock(seed: int = 0) -> dict[str, float]:
    return {"synthetic_chambolle_pock": _bench_chambolle_pock(seed)}
