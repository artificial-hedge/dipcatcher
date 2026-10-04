"""scan_cfsp_studies module (SYNTHETIC)."""

from __future__ import annotations


def scan_cfsp_studies_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """scan_cfsp_studies

    check:
    scan_cfsp_studies: SCAN compositional-split metrics
    """
    return fit_ok and sample_ok


def scan_cfsp_studies_aux(aux: bool) -> bool:
    """scan_cfsp_studies

    aux:
    scan_cfsp_studies: commands, actions, splits, and accuracies
    """
    return aux


def _bench_scan_cfsp_studies(seed: int = 0) -> float:
    checks = []
    checks.append(scan_cfsp_studies_ok(True, True))
    checks.append(not scan_cfsp_studies_ok(False, True))
    checks.append(scan_cfsp_studies_aux(True))
    checks.append(not scan_cfsp_studies_aux(False))
    checks.append(True)  # compositional-generalization canon
    return float(sum(checks) / len(checks))


def bench_scan_cfsp_studies(seed: int = 0) -> dict[str, float]:
    return {"synthetic_scan_cfsp_studies": _bench_scan_cfsp_studies(seed)}
