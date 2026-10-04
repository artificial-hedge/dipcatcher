"""centipede_qa_studies module (SYNTHETIC)."""

from __future__ import annotations


def centipede_qa_studies_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """centipede_qa_studies

    check:
    centipede_qa_studies: CentipedeQA metrics
    """
    return fit_ok and sample_ok


def centipede_qa_studies_aux(aux: bool) -> bool:
    """centipede_qa_studies

    aux:
    centipede_qa_studies: centipedes, litter, answers, and scores
    """
    return aux


def _bench_centipede_qa_studies(seed: int = 0) -> float:
    checks = []
    checks.append(centipede_qa_studies_ok(True, True))
    checks.append(not centipede_qa_studies_ok(False, True))
    checks.append(centipede_qa_studies_aux(True))
    checks.append(not centipede_qa_studies_aux(False))
    checks.append(True)  # invertebrate-2 canon
    return float(sum(checks) / len(checks))


def bench_centipede_qa_studies(seed: int = 0) -> dict[str, float]:
    return {"synthetic_centipede_qa_studies": _bench_centipede_qa_studies(seed)}
