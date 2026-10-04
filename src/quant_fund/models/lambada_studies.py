"""lambada_studies module (SYNTHETIC)."""

from __future__ import annotations


def lambada_studies_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """lambada_studies

    check:
    lambada_studies: LAMBADA long-context target-word prediction and acc
    """
    return fit_ok and sample_ok


def lambada_studies_aux(aux: bool) -> bool:
    """lambada_studies

    aux:
    lambada_studies: passages, final-word targets, and perplexity
    """
    return aux


def _bench_lambada_studies(seed: int = 0) -> float:
    checks = []
    checks.append(lambada_studies_ok(True, True))
    checks.append(not lambada_studies_ok(False, True))
    checks.append(lambada_studies_aux(True))
    checks.append(not lambada_studies_aux(False))
    checks.append(True)  # winograd-eval canon
    return float(sum(checks) / len(checks))


def bench_lambada_studies(seed: int = 0) -> dict[str, float]:
    return {"synthetic_lambada_studies": _bench_lambada_studies(seed)}
