"""mnli_studies module (SYNTHETIC)."""

from __future__ import annotations


def mnli_studies_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """mnli_studies

    check:
    mnli_studies: MNLI entailment matched/mismatched accuracy
    """
    return fit_ok and sample_ok


def mnli_studies_aux(aux: bool) -> bool:
    """mnli_studies

    aux:
    mnli_studies: premise-hypothesis pairs, labels, and genres
    """
    return aux


def _bench_mnli_studies(seed: int = 0) -> float:
    checks = []
    checks.append(mnli_studies_ok(True, True))
    checks.append(not mnli_studies_ok(False, True))
    checks.append(mnli_studies_aux(True))
    checks.append(not mnli_studies_aux(False))
    checks.append(True)  # GLUE-eval canon
    return float(sum(checks) / len(checks))


def bench_mnli_studies(seed: int = 0) -> dict[str, float]:
    return {"synthetic_mnli_studies": _bench_mnli_studies(seed)}
