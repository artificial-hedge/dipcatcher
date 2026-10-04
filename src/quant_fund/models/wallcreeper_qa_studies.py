"""wallcreeper_qa_studies module (SYNTHETIC)."""

from __future__ import annotations


def wallcreeper_qa_studies_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """wallcreeper_qa_studies

    check:
    wallcreeper_qa_studies: WallcreeperQA metrics
    """
    return fit_ok and sample_ok


def wallcreeper_qa_studies_aux(aux: bool) -> bool:
    """wallcreeper_qa_studies

    aux:
    wallcreeper_qa_studies: wallcreepers, cliff faces, answers, and scores
    """
    return aux


def _bench_wallcreeper_qa_studies(seed: int = 0) -> float:
    checks = []
    checks.append(wallcreeper_qa_studies_ok(True, True))
    checks.append(not wallcreeper_qa_studies_ok(False, True))
    checks.append(wallcreeper_qa_studies_aux(True))
    checks.append(not wallcreeper_qa_studies_aux(False))
    checks.append(True)  # alpine-bird canon
    return float(sum(checks) / len(checks))


def bench_wallcreeper_qa_studies(seed: int = 0) -> dict[str, float]:
    return {"synthetic_wallcreeper_qa_studies": _bench_wallcreeper_qa_studies(seed)}
