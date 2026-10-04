"""wildguard_studies module (SYNTHETIC)."""

from __future__ import annotations


def wildguard_studies_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """wildguard_studies

    check:
    wildguard_studies: WildGuard prompt/response harm detection and F1
    """
    return fit_ok and sample_ok


def wildguard_studies_aux(aux: bool) -> bool:
    """wildguard_studies

    aux:
    wildguard_studies: adversarial+vanilla items, labels, and metrics
    """
    return aux


def _bench_wildguard_studies(seed: int = 0) -> float:
    checks = []
    checks.append(wildguard_studies_ok(True, True))
    checks.append(not wildguard_studies_ok(False, True))
    checks.append(wildguard_studies_aux(True))
    checks.append(not wildguard_studies_aux(False))
    checks.append(True)  # safety-benchmark canon
    return float(sum(checks) / len(checks))


def bench_wildguard_studies(seed: int = 0) -> dict[str, float]:
    return {"synthetic_wildguard_studies": _bench_wildguard_studies(seed)}
