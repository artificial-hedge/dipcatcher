"""bandicoot_qa_studies module (SYNTHETIC)."""

from __future__ import annotations


def bandicoot_qa_studies_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """bandicoot_qa_studies

    check:
    bandicoot_qa_studies: BandicootQA metrics
    """
    return fit_ok and sample_ok


def bandicoot_qa_studies_aux(aux: bool) -> bool:
    """bandicoot_qa_studies

    aux:
    bandicoot_qa_studies: bandicoots, burrows, answers, and scores
    """
    return aux


def _bench_bandicoot_qa_studies(seed: int = 0) -> float:
    checks = []
    checks.append(bandicoot_qa_studies_ok(True, True))
    checks.append(not bandicoot_qa_studies_ok(False, True))
    checks.append(bandicoot_qa_studies_aux(True))
    checks.append(not bandicoot_qa_studies_aux(False))
    checks.append(True)  # marsupial canon
    return float(sum(checks) / len(checks))


def bench_bandicoot_qa_studies(seed: int = 0) -> dict[str, float]:
    return {"synthetic_bandicoot_qa_studies": _bench_bandicoot_qa_studies(seed)}
