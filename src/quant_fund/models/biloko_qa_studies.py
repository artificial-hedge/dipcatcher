"""biloko_qa_studies module (SYNTHETIC)."""

from __future__ import annotations


def biloko_qa_studies_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """biloko_qa_studies

    check:
    biloko_qa_studies: BilokoQA metrics
    """
    return fit_ok and sample_ok


def biloko_qa_studies_aux(aux: bool) -> bool:
    """biloko_qa_studies

    aux:
    biloko_qa_studies: bilokos, hollow guards, answers, and scores
    """
    return aux


def _bench_biloko_qa_studies(seed: int = 0) -> float:
    checks = []
    checks.append(biloko_qa_studies_ok(True, True))
    checks.append(not biloko_qa_studies_ok(False, True))
    checks.append(biloko_qa_studies_aux(True))
    checks.append(not biloko_qa_studies_aux(False))
    checks.append(True)  # african-beast canon
    return float(sum(checks) / len(checks))


def bench_biloko_qa_studies(seed: int = 0) -> dict[str, float]:
    return {"synthetic_biloko_qa_studies": _bench_biloko_qa_studies(seed)}
