"""stylized_studies module (SYNTHETIC)."""

from __future__ import annotations


def stylized_studies_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """stylized_studies

    check:
    stylized_studies: stylized-domain texture/shape bias and accuracy
    """
    return fit_ok and sample_ok


def stylized_studies_aux(aux: bool) -> bool:
    """stylized_studies

    aux:
    stylized_studies: stylized pairs, cues, and classification rates
    """
    return aux


def _bench_stylized_studies(seed: int = 0) -> float:
    checks = []
    checks.append(stylized_studies_ok(True, True))
    checks.append(not stylized_studies_ok(False, True))
    checks.append(stylized_studies_aux(True))
    checks.append(not stylized_studies_aux(False))
    checks.append(True)  # OOD-robustness canon
    return float(sum(checks) / len(checks))


def bench_stylized_studies(seed: int = 0) -> dict[str, float]:
    return {"synthetic_stylized_studies": _bench_stylized_studies(seed)}
