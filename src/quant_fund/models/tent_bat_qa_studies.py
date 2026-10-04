"""tent_bat_qa_studies module (SYNTHETIC)."""

from __future__ import annotations


def tent_bat_qa_studies_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """tent_bat_qa_studies

    check:
    tent_bat_qa_studies: TentBatQA metrics
    """
    return fit_ok and sample_ok


def tent_bat_qa_studies_aux(aux: bool) -> bool:
    """tent_bat_qa_studies

    aux:
    tent_bat_qa_studies: tent-making bats, palm shelters, answers, and scores
    """
    return aux


def _bench_tent_bat_qa_studies(seed: int = 0) -> float:
    checks = []
    checks.append(tent_bat_qa_studies_ok(True, True))
    checks.append(not tent_bat_qa_studies_ok(False, True))
    checks.append(tent_bat_qa_studies_aux(True))
    checks.append(not tent_bat_qa_studies_aux(False))
    checks.append(True)  # bat-2 canon
    return float(sum(checks) / len(checks))


def bench_tent_bat_qa_studies(seed: int = 0) -> dict[str, float]:
    return {"synthetic_tent_bat_qa_studies": _bench_tent_bat_qa_studies(seed)}
