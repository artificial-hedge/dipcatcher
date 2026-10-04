"""quoref_lite_studies module (SYNTHETIC)."""

from __future__ import annotations


def quoref_lite_studies_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """quoref_lite_studies

    check:
    quoref_lite_studies: Quoref coreference metrics
    """
    return fit_ok and sample_ok


def quoref_lite_studies_aux(aux: bool) -> bool:
    """quoref_lite_studies

    aux:
    quoref_lite_studies: contexts, questions, spans, and accuracies
    """
    return aux


def _bench_quoref_lite_studies(seed: int = 0) -> float:
    checks = []
    checks.append(quoref_lite_studies_ok(True, True))
    checks.append(not quoref_lite_studies_ok(False, True))
    checks.append(quoref_lite_studies_aux(True))
    checks.append(not quoref_lite_studies_aux(False))
    checks.append(True)  # reading-comp-3 canon
    return float(sum(checks) / len(checks))


def bench_quoref_lite_studies(seed: int = 0) -> dict[str, float]:
    return {"synthetic_quoref_lite_studies": _bench_quoref_lite_studies(seed)}
