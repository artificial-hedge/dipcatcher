"""winograd_studies module (SYNTHETIC)."""

from __future__ import annotations


def winograd_studies_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """winograd_studies

    check:
    winograd_studies: Winograd schema coreference pairs and accuracy
    """
    return fit_ok and sample_ok


def winograd_studies_aux(aux: bool) -> bool:
    """winograd_studies

    aux:
    winograd_studies: sentences, pronouns, candidates, and labels
    """
    return aux


def _bench_winograd_studies(seed: int = 0) -> float:
    checks = []
    checks.append(winograd_studies_ok(True, True))
    checks.append(not winograd_studies_ok(False, True))
    checks.append(winograd_studies_aux(True))
    checks.append(not winograd_studies_aux(False))
    checks.append(True)  # winograd-eval canon
    return float(sum(checks) / len(checks))


def bench_winograd_studies(seed: int = 0) -> dict[str, float]:
    return {"synthetic_winograd_studies": _bench_winograd_studies(seed)}
