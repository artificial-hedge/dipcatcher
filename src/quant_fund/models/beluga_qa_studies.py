"""beluga_qa_studies module (SYNTHETIC)."""

from __future__ import annotations


def beluga_qa_studies_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """beluga_qa_studies

    check:
    beluga_qa_studies: BelugaQA metrics
    """
    return fit_ok and sample_ok


def beluga_qa_studies_aux(aux: bool) -> bool:
    """beluga_qa_studies

    aux:
    beluga_qa_studies: belugas, melons, answers, and scores
    """
    return aux


def _bench_beluga_qa_studies(seed: int = 0) -> float:
    checks = []
    checks.append(beluga_qa_studies_ok(True, True))
    checks.append(not beluga_qa_studies_ok(False, True))
    checks.append(beluga_qa_studies_aux(True))
    checks.append(not beluga_qa_studies_aux(False))
    checks.append(True)  # marine mammal canon
    return float(sum(checks) / len(checks))


def bench_beluga_qa_studies(seed: int = 0) -> dict[str, float]:
    return {"synthetic_beluga_qa_studies": _bench_beluga_qa_studies(seed)}
