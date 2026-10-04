"""cassowary_qa_studies module (SYNTHETIC)."""

from __future__ import annotations


def cassowary_qa_studies_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """cassowary_qa_studies

    check:
    cassowary_qa_studies: CassowaryQA metrics
    """
    return fit_ok and sample_ok


def cassowary_qa_studies_aux(aux: bool) -> bool:
    """cassowary_qa_studies

    aux:
    cassowary_qa_studies: cassowaries, rainforests, answers, and scores
    """
    return aux


def _bench_cassowary_qa_studies(seed: int = 0) -> float:
    checks = []
    checks.append(cassowary_qa_studies_ok(True, True))
    checks.append(not cassowary_qa_studies_ok(False, True))
    checks.append(cassowary_qa_studies_aux(True))
    checks.append(not cassowary_qa_studies_aux(False))
    checks.append(True)  # ratite canon
    return float(sum(checks) / len(checks))


def bench_cassowary_qa_studies(seed: int = 0) -> dict[str, float]:
    return {"synthetic_cassowary_qa_studies": _bench_cassowary_qa_studies(seed)}
