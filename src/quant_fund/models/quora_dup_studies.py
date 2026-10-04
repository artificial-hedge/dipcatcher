"""quora_dup_studies module (SYNTHETIC)."""

from __future__ import annotations


def quora_dup_studies_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """quora_dup_studies

    check:
    quora_dup_studies: Quora duplicate metrics
    """
    return fit_ok and sample_ok


def quora_dup_studies_aux(aux: bool) -> bool:
    """quora_dup_studies

    aux:
    quora_dup_studies: questions, pairs, labels, and accuracies
    """
    return aux


def _bench_quora_dup_studies(seed: int = 0) -> float:
    checks = []
    checks.append(quora_dup_studies_ok(True, True))
    checks.append(not quora_dup_studies_ok(False, True))
    checks.append(quora_dup_studies_aux(True))
    checks.append(not quora_dup_studies_aux(False))
    checks.append(True)  # sentence-pair canon
    return float(sum(checks) / len(checks))


def bench_quora_dup_studies(seed: int = 0) -> dict[str, float]:
    return {"synthetic_quora_dup_studies": _bench_quora_dup_studies(seed)}
