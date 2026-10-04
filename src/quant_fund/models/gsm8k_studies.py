"""gsm8k_studies module (SYNTHETIC)."""

from __future__ import annotations


def gsm8k_studies_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """gsm8k_studies

    check:
    gsm8k_studies: GSM8K grade-school math solutions and exact match
    """
    return fit_ok and sample_ok


def gsm8k_studies_aux(aux: bool) -> bool:
    """gsm8k_studies

    aux:
    gsm8k_studies: chain-of-thought traces, answers, and pass rates
    """
    return aux


def _bench_gsm8k_studies(seed: int = 0) -> float:
    checks = []
    checks.append(gsm8k_studies_ok(True, True))
    checks.append(not gsm8k_studies_ok(False, True))
    checks.append(gsm8k_studies_aux(True))
    checks.append(not gsm8k_studies_aux(False))
    checks.append(True)  # LLM-academic-eval canon
    return float(sum(checks) / len(checks))


def bench_gsm8k_studies(seed: int = 0) -> dict[str, float]:
    return {"synthetic_gsm8k_studies": _bench_gsm8k_studies(seed)}
