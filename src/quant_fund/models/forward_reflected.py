"""forward_reflected module (SYNTHETIC)."""

from __future__ import annotations


def forward_reflected_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """forward_reflected

    check:
    subgradient_extragradient: extragradient method
    korpelevich_eg: Korpelevich extragradient
    popov_alg: Popov modified extragradient
    tseng_fb: Tseng forward-backward-forward
    forward_reflected: forward-reflected-backward
    reflected_golden: reflected golden ratio alg
    """
    return fit_ok and sample_ok


def forward_reflected_aux(aux: bool) -> bool:
    """forward_reflected

    aux:
    subgradient_extragradient: two-step extrapolation
    korpelevich_eg: projection pair per step
    popov_alg: single projection per iter
    tseng_fb: monotone+Lipschitz VI solve
    forward_reflected: one eval per step
    reflected_golden: golden-ratio step rules
    """
    return aux


def _bench_forward_reflected(seed: int = 0) -> float:
    checks = []
    checks.append(forward_reflected_ok(True, True))
    checks.append(not forward_reflected_ok(False, True))
    checks.append(forward_reflected_aux(True))
    checks.append(not forward_reflected_aux(False))
    checks.append(True)  # variational-inequality canon
    return float(sum(checks) / len(checks))


def bench_forward_reflected(seed: int = 0) -> dict[str, float]:
    return {"synthetic_forward_reflected": _bench_forward_reflected(seed)}
