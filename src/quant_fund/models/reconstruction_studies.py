"""reconstruction_studies module (SYNTHETIC)."""

from __future__ import annotations


def reconstruction_studies_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """reconstruction_studies

    check:
    reconstruction_studies: gradient-inversion sample reconstruction fidelity
    """
    return fit_ok and sample_ok


def reconstruction_studies_aux(aux: bool) -> bool:
    """reconstruction_studies

    aux:
    reconstruction_studies: gradients, optimized inputs, and PSNR scores
    """
    return aux


def _bench_reconstruction_studies(seed: int = 0) -> float:
    checks = []
    checks.append(reconstruction_studies_ok(True, True))
    checks.append(not reconstruction_studies_ok(False, True))
    checks.append(reconstruction_studies_aux(True))
    checks.append(not reconstruction_studies_aux(False))
    checks.append(True)  # privacy-inference canon
    return float(sum(checks) / len(checks))


def bench_reconstruction_studies(seed: int = 0) -> dict[str, float]:
    return {"synthetic_reconstruction_studies": _bench_reconstruction_studies(seed)}
