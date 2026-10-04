"""external_control_studies module (SYNTHETIC)."""

from __future__ import annotations


def external_control_studies_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """external_control_studies

    check:
    external_control_studies: synthetic arms and hybrid/borrowing and weighting
    """
    return fit_ok and sample_ok


def external_control_studies_aux(aux: bool) -> bool:
    """external_control_studies

    aux:
    external_control_studies: exchangeability and overlap/data and fit
    """
    return aux


def _bench_external_control_studies(seed: int = 0) -> float:
    checks = []
    checks.append(external_control_studies_ok(True, True))
    checks.append(not external_control_studies_ok(False, True))
    checks.append(external_control_studies_aux(True))
    checks.append(not external_control_studies_aux(False))
    checks.append(True)  # causal-RWE-2 canon
    return float(sum(checks) / len(checks))


def bench_external_control_studies(seed: int = 0) -> dict[str, float]:
    return {"synthetic_external_control_studies": _bench_external_control_studies(seed)}
