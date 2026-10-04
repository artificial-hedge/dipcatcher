"""trojan_studies module (SYNTHETIC)."""

from __future__ import annotations


def trojan_studies_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """trojan_studies

    check:
    trojan_studies: trojaned models/neurons and anomaly indices
    """
    return fit_ok and sample_ok


def trojan_studies_aux(aux: bool) -> bool:
    """trojan_studies

    aux:
    trojan_studies: trojan scans/triggers and detection rates
    """
    return aux


def _bench_trojan_studies(seed: int = 0) -> float:
    checks = []
    checks.append(trojan_studies_ok(True, True))
    checks.append(not trojan_studies_ok(False, True))
    checks.append(trojan_studies_aux(True))
    checks.append(not trojan_studies_aux(False))
    checks.append(True)  # backdoor-eval canon
    return float(sum(checks) / len(checks))


def bench_trojan_studies(seed: int = 0) -> dict[str, float]:
    return {"synthetic_trojan_studies": _bench_trojan_studies(seed)}
