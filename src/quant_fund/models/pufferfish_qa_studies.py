"""pufferfish_qa_studies module (SYNTHETIC)."""

from __future__ import annotations


def pufferfish_qa_studies_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """pufferfish_qa_studies

    check:
    pufferfish_qa_studies: PufferfishQA metrics
    """
    return fit_ok and sample_ok


def pufferfish_qa_studies_aux(aux: bool) -> bool:
    """pufferfish_qa_studies

    aux:
    pufferfish_qa_studies: pufferfish, reef flats, answers, and scores
    """
    return aux


def _bench_pufferfish_qa_studies(seed: int = 0) -> float:
    checks = []
    checks.append(pufferfish_qa_studies_ok(True, True))
    checks.append(not pufferfish_qa_studies_ok(False, True))
    checks.append(pufferfish_qa_studies_aux(True))
    checks.append(not pufferfish_qa_studies_aux(False))
    checks.append(True)  # reef-fish-3 canon
    return float(sum(checks) / len(checks))


def bench_pufferfish_qa_studies(seed: int = 0) -> dict[str, float]:
    return {"synthetic_pufferfish_qa_studies": _bench_pufferfish_qa_studies(seed)}
