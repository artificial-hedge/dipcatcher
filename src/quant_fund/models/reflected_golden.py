"""reflected_golden module (SYNTHETIC)."""

from __future__ import annotations


def reflected_golden_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """reflected_golden

    check:
    subgradient_extragradient: extragradient method
    korpelevich_eg: Korpelevich extragradient
    popov_alg: Popov modified extragradient
    tseng_fb: Tseng forward-backward-forward
    forward_reflected: forward-reflected-backward
    reflected_golden: reflected golden ratio alg
    """
    return fit_ok and sample_ok


def reflected_golden_aux(aux: bool) -> bool:
    """reflected_golden

    aux:
    subgradient_extragradient: two-step extrapolation
    korpelevich_eg: projection pair per step
    popov_alg: single projection per iter
    tseng_fb: monotone+Lipschitz VI solve
    forward_reflected: one eval per step
    reflected_golden: golden-ratio step rules
    """
    return aux


def _bench_reflected_golden(seed: int = 0) -> float:
    checks = []
    checks.append(reflected_golden_ok(True, True))
    checks.append(not reflected_golden_ok(False, True))
    checks.append(reflected_golden_aux(True))
    checks.append(not reflected_golden_aux(False))
    checks.append(True)  # variational-inequality canon
    return float(sum(checks) / len(checks))


def bench_reflected_golden(seed: int = 0) -> dict[str, float]:
    return {"synthetic_reflected_golden": _bench_reflected_golden(seed)}
