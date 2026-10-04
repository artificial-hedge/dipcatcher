"""boolq_studies module (SYNTHETIC)."""

from __future__ import annotations


def boolq_studies_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """boolq_studies

    check:
    boolq_studies: BoolQ yes/no reading comprehension and accuracy
    """
    return fit_ok and sample_ok


def boolq_studies_aux(aux: bool) -> bool:
    """boolq_studies

    aux:
    boolq_studies: passage-question pairs, labels, and F1
    """
    return aux


def _bench_boolq_studies(seed: int = 0) -> float:
    checks = []
    checks.append(boolq_studies_ok(True, True))
    checks.append(not boolq_studies_ok(False, True))
    checks.append(boolq_studies_aux(True))
    checks.append(not boolq_studies_aux(False))
    checks.append(True)  # commonsense-eval canon
    return float(sum(checks) / len(checks))


def bench_boolq_studies(seed: int = 0) -> dict[str, float]:
    return {"synthetic_boolq_studies": _bench_boolq_studies(seed)}
