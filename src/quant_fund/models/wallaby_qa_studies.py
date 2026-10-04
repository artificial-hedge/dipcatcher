"""wallaby_qa_studies module (SYNTHETIC)."""

from __future__ import annotations


def wallaby_qa_studies_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """wallaby_qa_studies

    check:
    wallaby_qa_studies: WallabyQA metrics
    """
    return fit_ok and sample_ok


def wallaby_qa_studies_aux(aux: bool) -> bool:
    """wallaby_qa_studies

    aux:
    wallaby_qa_studies: wallabies, pouches, answers, and scores
    """
    return aux


def _bench_wallaby_qa_studies(seed: int = 0) -> float:
    checks = []
    checks.append(wallaby_qa_studies_ok(True, True))
    checks.append(not wallaby_qa_studies_ok(False, True))
    checks.append(wallaby_qa_studies_aux(True))
    checks.append(not wallaby_qa_studies_aux(False))
    checks.append(True)  # marsupial canon
    return float(sum(checks) / len(checks))


def bench_wallaby_qa_studies(seed: int = 0) -> dict[str, float]:
    return {"synthetic_wallaby_qa_studies": _bench_wallaby_qa_studies(seed)}
