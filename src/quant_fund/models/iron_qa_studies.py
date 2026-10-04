"""iron_qa_studies module (SYNTHETIC)."""

from __future__ import annotations


def iron_qa_studies_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """iron_qa_studies

    check:
    iron_qa_studies: IronQA metrics
    """
    return fit_ok and sample_ok


def iron_qa_studies_aux(aux: bool) -> bool:
    """iron_qa_studies

    aux:
    iron_qa_studies: irons, ores, answers, and scores
    """
    return aux


def _bench_iron_qa_studies(seed: int = 0) -> float:
    checks = []
    checks.append(iron_qa_studies_ok(True, True))
    checks.append(not iron_qa_studies_ok(False, True))
    checks.append(iron_qa_studies_aux(True))
    checks.append(not iron_qa_studies_aux(False))
    checks.append(True)  # material canon
    return float(sum(checks) / len(checks))


def bench_iron_qa_studies(seed: int = 0) -> dict[str, float]:
    return {"synthetic_iron_qa_studies": _bench_iron_qa_studies(seed)}
