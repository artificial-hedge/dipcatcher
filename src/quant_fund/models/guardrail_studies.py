"""guardrail_studies module (SYNTHETIC)."""

from __future__ import annotations


def guardrail_studies_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """guardrail_studies

    check:
    guardrail_studies: input-output filters and policy enforcement/guards and classifiers
    """
    return fit_ok and sample_ok


def guardrail_studies_aux(aux: bool) -> bool:
    """guardrail_studies

    aux:
    guardrail_studies: constitution adherence and refusal calibration/coverage and false positives
    """
    return aux


def _bench_guardrail_studies(seed: int = 0) -> float:
    checks = []
    checks.append(guardrail_studies_ok(True, True))
    checks.append(not guardrail_studies_ok(False, True))
    checks.append(guardrail_studies_aux(True))
    checks.append(not guardrail_studies_aux(False))
    checks.append(True)  # AI-safety canon
    return float(sum(checks) / len(checks))


def bench_guardrail_studies(seed: int = 0) -> dict[str, float]:
    return {"synthetic_guardrail_studies": _bench_guardrail_studies(seed)}
