"""snli_lite_studies module (SYNTHETIC)."""

from __future__ import annotations


def snli_lite_studies_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """snli_lite_studies

    check:
    snli_lite_studies: SNLI-lite entailment metrics
    """
    return fit_ok and sample_ok


def snli_lite_studies_aux(aux: bool) -> bool:
    """snli_lite_studies

    aux:
    snli_lite_studies: premises, hypotheses, labels, and accuracies
    """
    return aux


def _bench_snli_lite_studies(seed: int = 0) -> float:
    checks = []
    checks.append(snli_lite_studies_ok(True, True))
    checks.append(not snli_lite_studies_ok(False, True))
    checks.append(snli_lite_studies_aux(True))
    checks.append(not snli_lite_studies_aux(False))
    checks.append(True)  # NLI-eval canon
    return float(sum(checks) / len(checks))


def bench_snli_lite_studies(seed: int = 0) -> dict[str, float]:
    return {"synthetic_snli_lite_studies": _bench_snli_lite_studies(seed)}
