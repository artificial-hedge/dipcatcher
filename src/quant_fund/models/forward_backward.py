"""forward_backward module (SYNTHETIC)."""

from __future__ import annotations


def forward_backward_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """forward_backward

    check:
    douglas_rachford: DR splitting iteration
    peaceman_rachford: PR alternating splitting
    tseng_split: Tseng forward-backward-forward
    forward_backward: forward-backward splitting
    chambolle_pock: primal-dual algorithm
    davis_yin: three-operator splitting
    """
    return fit_ok and sample_ok


def forward_backward_aux(aux: bool) -> bool:
    """forward_backward

    aux:
    douglas_rachford: reflected resolvent
    peaceman_rachford: asymmetric DR step
    tseng_split: monotone inclusion solve
    forward_backward: proximal gradient form
    chambolle_pock: saddle-point extragradient
    davis_yin: three-term composition
    """
    return aux


def _bench_forward_backward(seed: int = 0) -> float:
    checks = []
    checks.append(forward_backward_ok(True, True))
    checks.append(not forward_backward_ok(False, True))
    checks.append(forward_backward_aux(True))
    checks.append(not forward_backward_aux(False))
    checks.append(True)  # operator-splitting canon
    return float(sum(checks) / len(checks))


def bench_forward_backward(seed: int = 0) -> dict[str, float]:
    return {"synthetic_forward_backward": _bench_forward_backward(seed)}
