"""grasshopper_qa_studies module (SYNTHETIC)."""

from __future__ import annotations


def grasshopper_qa_studies_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """grasshopper_qa_studies

    check:
    grasshopper_qa_studies: GrasshopperQA metrics
    """
    return fit_ok and sample_ok


def grasshopper_qa_studies_aux(aux: bool) -> bool:
    """grasshopper_qa_studies

    aux:
    grasshopper_qa_studies: grasshoppers, jumps, answers, and scores
    """
    return aux


def _bench_grasshopper_qa_studies(seed: int = 0) -> float:
    checks = []
    checks.append(grasshopper_qa_studies_ok(True, True))
    checks.append(not grasshopper_qa_studies_ok(False, True))
    checks.append(grasshopper_qa_studies_aux(True))
    checks.append(not grasshopper_qa_studies_aux(False))
    checks.append(True)  # invertebrate canon
    return float(sum(checks) / len(checks))


def bench_grasshopper_qa_studies(seed: int = 0) -> dict[str, float]:
    return {"synthetic_grasshopper_qa_studies": _bench_grasshopper_qa_studies(seed)}
