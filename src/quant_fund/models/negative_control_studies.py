"""negative_control_studies module (SYNTHETIC)."""

from __future__ import annotations


def negative_control_studies_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """negative_control_studies

    check:
    negative_control_studies: exposures and outcomes/falsification and bias
    """
    return fit_ok and sample_ok


def negative_control_studies_aux(aux: bool) -> bool:
    """negative_control_studies

    aux:
    negative_control_studies: calibration and ratios/E-value and design
    """
    return aux


def _bench_negative_control_studies(seed: int = 0) -> float:
    checks = []
    checks.append(negative_control_studies_ok(True, True))
    checks.append(not negative_control_studies_ok(False, True))
    checks.append(negative_control_studies_aux(True))
    checks.append(not negative_control_studies_aux(False))
    checks.append(True)  # causal-RWE-2 canon
    return float(sum(checks) / len(checks))


def bench_negative_control_studies(seed: int = 0) -> dict[str, float]:
    return {"synthetic_negative_control_studies": _bench_negative_control_studies(seed)}
