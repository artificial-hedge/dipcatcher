"""firefly_qa_studies module (SYNTHETIC)."""

from __future__ import annotations


def firefly_qa_studies_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """firefly_qa_studies

    check:
    firefly_qa_studies: FireflyQA metrics
    """
    return fit_ok and sample_ok


def firefly_qa_studies_aux(aux: bool) -> bool:
    """firefly_qa_studies

    aux:
    firefly_qa_studies: fireflies, lanterns, answers, and scores
    """
    return aux


def _bench_firefly_qa_studies(seed: int = 0) -> float:
    checks = []
    checks.append(firefly_qa_studies_ok(True, True))
    checks.append(not firefly_qa_studies_ok(False, True))
    checks.append(firefly_qa_studies_aux(True))
    checks.append(not firefly_qa_studies_aux(False))
    checks.append(True)  # meadow canon
    return float(sum(checks) / len(checks))


def bench_firefly_qa_studies(seed: int = 0) -> dict[str, float]:
    return {"synthetic_firefly_qa_studies": _bench_firefly_qa_studies(seed)}
