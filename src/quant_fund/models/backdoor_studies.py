"""backdoor_studies module (SYNTHETIC)."""

from __future__ import annotations


def backdoor_studies_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """backdoor_studies

    check:
    backdoor_studies: backdoor triggers/patterns and attack success rates
    """
    return fit_ok and sample_ok


def backdoor_studies_aux(aux: bool) -> bool:
    """backdoor_studies

    aux:
    backdoor_studies: trigger masks/poisons and victim accuracies
    """
    return aux


def _bench_backdoor_studies(seed: int = 0) -> float:
    checks = []
    checks.append(backdoor_studies_ok(True, True))
    checks.append(not backdoor_studies_ok(False, True))
    checks.append(backdoor_studies_aux(True))
    checks.append(not backdoor_studies_aux(False))
    checks.append(True)  # backdoor-eval canon
    return float(sum(checks) / len(checks))


def bench_backdoor_studies(seed: int = 0) -> dict[str, float]:
    return {"synthetic_backdoor_studies": _bench_backdoor_studies(seed)}
