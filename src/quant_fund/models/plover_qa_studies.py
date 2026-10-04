"""plover_qa_studies module (SYNTHETIC)."""

from __future__ import annotations


def plover_qa_studies_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """plover_qa_studies

    check:
    plover_qa_studies: PloverQA metrics
    """
    return fit_ok and sample_ok


def plover_qa_studies_aux(aux: bool) -> bool:
    """plover_qa_studies

    aux:
    plover_qa_studies: plovers, shorelines, answers, and scores
    """
    return aux


def _bench_plover_qa_studies(seed: int = 0) -> float:
    checks = []
    checks.append(plover_qa_studies_ok(True, True))
    checks.append(not plover_qa_studies_ok(False, True))
    checks.append(plover_qa_studies_aux(True))
    checks.append(not plover_qa_studies_aux(False))
    checks.append(True)  # shorebird canon
    return float(sum(checks) / len(checks))


def bench_plover_qa_studies(seed: int = 0) -> dict[str, float]:
    return {"synthetic_plover_qa_studies": _bench_plover_qa_studies(seed)}
