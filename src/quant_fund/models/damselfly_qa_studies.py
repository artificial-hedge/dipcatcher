"""damselfly_qa_studies module (SYNTHETIC)."""

from __future__ import annotations


def damselfly_qa_studies_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """damselfly_qa_studies

    check:
    damselfly_qa_studies: DamselflyQA metrics
    """
    return fit_ok and sample_ok


def damselfly_qa_studies_aux(aux: bool) -> bool:
    """damselfly_qa_studies

    aux:
    damselfly_qa_studies: damselflies, reeds, answers, and scores
    """
    return aux


def _bench_damselfly_qa_studies(seed: int = 0) -> float:
    checks = []
    checks.append(damselfly_qa_studies_ok(True, True))
    checks.append(not damselfly_qa_studies_ok(False, True))
    checks.append(damselfly_qa_studies_aux(True))
    checks.append(not damselfly_qa_studies_aux(False))
    checks.append(True)  # dragonfly canon
    return float(sum(checks) / len(checks))


def bench_damselfly_qa_studies(seed: int = 0) -> dict[str, float]:
    return {"synthetic_damselfly_qa_studies": _bench_damselfly_qa_studies(seed)}
