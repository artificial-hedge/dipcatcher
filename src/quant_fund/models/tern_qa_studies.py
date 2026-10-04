"""tern_qa_studies module (SYNTHETIC)."""

from __future__ import annotations


def tern_qa_studies_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """tern_qa_studies

    check:
    tern_qa_studies: TernQA metrics
    """
    return fit_ok and sample_ok


def tern_qa_studies_aux(aux: bool) -> bool:
    """tern_qa_studies

    aux:
    tern_qa_studies: terns, dives, answers, and scores
    """
    return aux


def _bench_tern_qa_studies(seed: int = 0) -> float:
    checks = []
    checks.append(tern_qa_studies_ok(True, True))
    checks.append(not tern_qa_studies_ok(False, True))
    checks.append(tern_qa_studies_aux(True))
    checks.append(not tern_qa_studies_aux(False))
    checks.append(True)  # shorebird canon
    return float(sum(checks) / len(checks))


def bench_tern_qa_studies(seed: int = 0) -> dict[str, float]:
    return {"synthetic_tern_qa_studies": _bench_tern_qa_studies(seed)}
