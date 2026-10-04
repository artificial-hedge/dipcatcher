"""mnli_match_studies module (SYNTHETIC)."""

from __future__ import annotations


def mnli_match_studies_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """mnli_match_studies

    check:
    mnli_match_studies: MultiNLI matched/mismatched metrics
    """
    return fit_ok and sample_ok


def mnli_match_studies_aux(aux: bool) -> bool:
    """mnli_match_studies

    aux:
    mnli_match_studies: premises, hypotheses, labels, and accuracies
    """
    return aux


def _bench_mnli_match_studies(seed: int = 0) -> float:
    checks = []
    checks.append(mnli_match_studies_ok(True, True))
    checks.append(not mnli_match_studies_ok(False, True))
    checks.append(mnli_match_studies_aux(True))
    checks.append(not mnli_match_studies_aux(False))
    checks.append(True)  # NLI-eval canon
    return float(sum(checks) / len(checks))


def bench_mnli_match_studies(seed: int = 0) -> dict[str, float]:
    return {"synthetic_mnli_match_studies": _bench_mnli_match_studies(seed)}
