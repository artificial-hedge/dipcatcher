"""fen_qa_studies module (SYNTHETIC)."""

from __future__ import annotations


def fen_qa_studies_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """fen_qa_studies

    check:
    fen_qa_studies: FenQA metrics
    """
    return fit_ok and sample_ok


def fen_qa_studies_aux(aux: bool) -> bool:
    """fen_qa_studies

    aux:
    fen_qa_studies: fens, marshes, answers, and scores
    """
    return aux


def _bench_fen_qa_studies(seed: int = 0) -> float:
    checks = []
    checks.append(fen_qa_studies_ok(True, True))
    checks.append(not fen_qa_studies_ok(False, True))
    checks.append(fen_qa_studies_aux(True))
    checks.append(not fen_qa_studies_aux(False))
    checks.append(True)  # moorland canon
    return float(sum(checks) / len(checks))


def bench_fen_qa_studies(seed: int = 0) -> dict[str, float]:
    return {"synthetic_fen_qa_studies": _bench_fen_qa_studies(seed)}
