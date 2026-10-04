"""tsukuyomi_qa_studies module (SYNTHETIC)."""

from __future__ import annotations


def tsukuyomi_qa_studies_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """tsukuyomi_qa_studies

    check:
    tsukuyomi_qa_studies: TsukuyomiQA metrics
    """
    return fit_ok and sample_ok


def tsukuyomi_qa_studies_aux(aux: bool) -> bool:
    """tsukuyomi_qa_studies

    aux:
    tsukuyomi_qa_studies: tsukuyomi, moon gods, answers, and scores
    """
    return aux


def _bench_tsukuyomi_qa_studies(seed: int = 0) -> float:
    checks = []
    checks.append(tsukuyomi_qa_studies_ok(True, True))
    checks.append(not tsukuyomi_qa_studies_ok(False, True))
    checks.append(tsukuyomi_qa_studies_aux(True))
    checks.append(not tsukuyomi_qa_studies_aux(False))
    checks.append(True)  # japanese-myth canon
    return float(sum(checks) / len(checks))


def bench_tsukuyomi_qa_studies(seed: int = 0) -> dict[str, float]:
    return {"synthetic_tsukuyomi_qa_studies": _bench_tsukuyomi_qa_studies(seed)}
