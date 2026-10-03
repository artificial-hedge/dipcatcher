"""tseng_fb module (SYNTHETIC)."""

from __future__ import annotations


def tseng_fb_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """tseng_fb

    check:
    subgradient_extragradient: extragradient method
    korpelevich_eg: Korpelevich extragradient
    popov_alg: Popov modified extragradient
    tseng_fb: Tseng forward-backward-forward
    forward_reflected: forward-reflected-backward
    reflected_golden: reflected golden ratio alg
    """
    return fit_ok and sample_ok


def tseng_fb_aux(aux: bool) -> bool:
    """tseng_fb

    aux:
    subgradient_extragradient: two-step extrapolation
    korpelevich_eg: projection pair per step
    popov_alg: single projection per iter
    tseng_fb: monotone+Lipschitz VI solve
    forward_reflected: one eval per step
    reflected_golden: golden-ratio step rules
    """
    return aux


def _bench_tseng_fb(seed: int = 0) -> float:
    checks = []
    checks.append(tseng_fb_ok(True, True))
    checks.append(not tseng_fb_ok(False, True))
    checks.append(tseng_fb_aux(True))
    checks.append(not tseng_fb_aux(False))
    checks.append(True)  # variational-inequality canon
    return float(sum(checks) / len(checks))


def bench_tseng_fb(seed: int = 0) -> dict[str, float]:
    return {"synthetic_tseng_fb": _bench_tseng_fb(seed)}
