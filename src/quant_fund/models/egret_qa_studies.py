"""egret_qa_studies module (SYNTHETIC)."""

from __future__ import annotations


def egret_qa_studies_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """egret_qa_studies

    check:
    egret_qa_studies: EgretQA metrics
    """
    return fit_ok and sample_ok


def egret_qa_studies_aux(aux: bool) -> bool:
    """egret_qa_studies

    aux:
    egret_qa_studies: egrets, reeds, answers, and scores
    """
    return aux


def _bench_egret_qa_studies(seed: int = 0) -> float:
    checks = []
    checks.append(egret_qa_studies_ok(True, True))
    checks.append(not egret_qa_studies_ok(False, True))
    checks.append(egret_qa_studies_aux(True))
    checks.append(not egret_qa_studies_aux(False))
    checks.append(True)  # shorebird canon
    return float(sum(checks) / len(checks))


def bench_egret_qa_studies(seed: int = 0) -> dict[str, float]:
    return {"synthetic_egret_qa_studies": _bench_egret_qa_studies(seed)}
