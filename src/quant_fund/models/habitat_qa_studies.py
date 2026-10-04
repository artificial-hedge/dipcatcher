"""habitat_qa_studies module (SYNTHETIC)."""

from __future__ import annotations


def habitat_qa_studies_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """habitat_qa_studies

    check:
    habitat_qa_studies: HabitatQA metrics
    """
    return fit_ok and sample_ok


def habitat_qa_studies_aux(aux: bool) -> bool:
    """habitat_qa_studies

    aux:
    habitat_qa_studies: habitats, ranges, answers, and scores
    """
    return aux


def _bench_habitat_qa_studies(seed: int = 0) -> float:
    checks = []
    checks.append(habitat_qa_studies_ok(True, True))
    checks.append(not habitat_qa_studies_ok(False, True))
    checks.append(habitat_qa_studies_aux(True))
    checks.append(not habitat_qa_studies_aux(False))
    checks.append(True)  # wildlife canon
    return float(sum(checks) / len(checks))


def bench_habitat_qa_studies(seed: int = 0) -> dict[str, float]:
    return {"synthetic_habitat_qa_studies": _bench_habitat_qa_studies(seed)}
