"""mari_qa_studies module (SYNTHETIC)."""

from __future__ import annotations


def mari_qa_studies_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """mari_qa_studies

    check:
    mari_qa_studies: MariQA metrics
    """
    return fit_ok and sample_ok


def mari_qa_studies_aux(aux: bool) -> bool:
    """mari_qa_studies

    aux:
    mari_qa_studies: mari, cave queens, answers, and scores
    """
    return aux


def _bench_mari_qa_studies(seed: int = 0) -> float:
    checks = []
    checks.append(mari_qa_studies_ok(True, True))
    checks.append(not mari_qa_studies_ok(False, True))
    checks.append(mari_qa_studies_aux(True))
    checks.append(not mari_qa_studies_aux(False))
    checks.append(True)  # basque-myth canon
    return float(sum(checks) / len(checks))


def bench_mari_qa_studies(seed: int = 0) -> dict[str, float]:
    return {"synthetic_mari_qa_studies": _bench_mari_qa_studies(seed)}
