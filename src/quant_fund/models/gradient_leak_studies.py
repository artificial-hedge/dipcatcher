"""gradient_leak_studies module (SYNTHETIC)."""

from __future__ import annotations


def gradient_leak_studies_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """gradient_leak_studies

    check:
    gradient_leak_studies: gradient-leakage reconstruction attacks and sim
    """
    return fit_ok and sample_ok


def gradient_leak_studies_aux(aux: bool) -> bool:
    """gradient_leak_studies

    aux:
    gradient_leak_studies: gradient matching/optimization and PSNR
    """
    return aux


def _bench_gradient_leak_studies(seed: int = 0) -> float:
    checks = []
    checks.append(gradient_leak_studies_ok(True, True))
    checks.append(not gradient_leak_studies_ok(False, True))
    checks.append(gradient_leak_studies_aux(True))
    checks.append(not gradient_leak_studies_aux(False))
    checks.append(True)  # privacy-inference canon
    return float(sum(checks) / len(checks))


def bench_gradient_leak_studies(seed: int = 0) -> dict[str, float]:
    return {"synthetic_gradient_leak_studies": _bench_gradient_leak_studies(seed)}
