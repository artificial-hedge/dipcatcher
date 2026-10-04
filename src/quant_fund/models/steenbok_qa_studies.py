"""steenbok_qa_studies module (SYNTHETIC)."""

from __future__ import annotations


def steenbok_qa_studies_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """steenbok_qa_studies

    check:
    steenbok_qa_studies: SteenbokQA metrics
    """
    return fit_ok and sample_ok


def steenbok_qa_studies_aux(aux: bool) -> bool:
    """steenbok_qa_studies

    aux:
    steenbok_qa_studies: steenboks, kalahari pans, answers, and scores
    """
    return aux


def _bench_steenbok_qa_studies(seed: int = 0) -> float:
    checks = []
    checks.append(steenbok_qa_studies_ok(True, True))
    checks.append(not steenbok_qa_studies_ok(False, True))
    checks.append(steenbok_qa_studies_aux(True))
    checks.append(not steenbok_qa_studies_aux(False))
    checks.append(True)  # dwarf-antelope canon
    return float(sum(checks) / len(checks))


def bench_steenbok_qa_studies(seed: int = 0) -> dict[str, float]:
    return {"synthetic_steenbok_qa_studies": _bench_steenbok_qa_studies(seed)}
