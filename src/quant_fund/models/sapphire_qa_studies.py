"""sapphire_qa_studies module (SYNTHETIC)."""

from __future__ import annotations


def sapphire_qa_studies_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """sapphire_qa_studies

    check:
    sapphire_qa_studies: SapphireQA metrics
    """
    return fit_ok and sample_ok


def sapphire_qa_studies_aux(aux: bool) -> bool:
    """sapphire_qa_studies

    aux:
    sapphire_qa_studies: sapphires, canopies, answers, and scores
    """
    return aux


def _bench_sapphire_qa_studies(seed: int = 0) -> float:
    checks = []
    checks.append(sapphire_qa_studies_ok(True, True))
    checks.append(not sapphire_qa_studies_ok(False, True))
    checks.append(sapphire_qa_studies_aux(True))
    checks.append(not sapphire_qa_studies_aux(False))
    checks.append(True)  # hummingbird canon
    return float(sum(checks) / len(checks))


def bench_sapphire_qa_studies(seed: int = 0) -> dict[str, float]:
    return {"synthetic_sapphire_qa_studies": _bench_sapphire_qa_studies(seed)}
