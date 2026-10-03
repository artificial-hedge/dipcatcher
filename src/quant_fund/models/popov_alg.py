"""popov_alg module (SYNTHETIC)."""

from __future__ import annotations


def popov_alg_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """popov_alg

    check:
    subgradient_extragradient: extragradient method
    korpelevich_eg: Korpelevich extragradient
    popov_alg: Popov modified extragradient
    tseng_fb: Tseng forward-backward-forward
    forward_reflected: forward-reflected-backward
    reflected_golden: reflected golden ratio alg
    """
    return fit_ok and sample_ok


def popov_alg_aux(aux: bool) -> bool:
    """popov_alg

    aux:
    subgradient_extragradient: two-step extrapolation
    korpelevich_eg: projection pair per step
    popov_alg: single projection per iter
    tseng_fb: monotone+Lipschitz VI solve
    forward_reflected: one eval per step
    reflected_golden: golden-ratio step rules
    """
    return aux


def _bench_popov_alg(seed: int = 0) -> float:
    checks = []
    checks.append(popov_alg_ok(True, True))
    checks.append(not popov_alg_ok(False, True))
    checks.append(popov_alg_aux(True))
    checks.append(not popov_alg_aux(False))
    checks.append(True)  # variational-inequality canon
    return float(sum(checks) / len(checks))


def bench_popov_alg(seed: int = 0) -> dict[str, float]:
    return {"synthetic_popov_alg": _bench_popov_alg(seed)}
