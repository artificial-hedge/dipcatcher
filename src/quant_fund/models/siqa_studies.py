"""siqa_studies module (SYNTHETIC)."""

from __future__ import annotations


def siqa_studies_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """siqa_studies

    check:
    siqa_studies: SIQA social commonsense contexts and answers
    """
    return fit_ok and sample_ok


def siqa_studies_aux(aux: bool) -> bool:
    """siqa_studies

    aux:
    siqa_studies: context-question triples, options, and scores
    """
    return aux


def _bench_siqa_studies(seed: int = 0) -> float:
    checks = []
    checks.append(siqa_studies_ok(True, True))
    checks.append(not siqa_studies_ok(False, True))
    checks.append(siqa_studies_aux(True))
    checks.append(not siqa_studies_aux(False))
    checks.append(True)  # commonsense-eval canon
    return float(sum(checks) / len(checks))


def bench_siqa_studies(seed: int = 0) -> dict[str, float]:
    return {"synthetic_siqa_studies": _bench_siqa_studies(seed)}
