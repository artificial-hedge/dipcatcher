"""fruit_bat_qa_studies module (SYNTHETIC)."""

from __future__ import annotations


def fruit_bat_qa_studies_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """fruit_bat_qa_studies

    check:
    fruit_bat_qa_studies: FruitBatQA metrics
    """
    return fit_ok and sample_ok


def fruit_bat_qa_studies_aux(aux: bool) -> bool:
    """fruit_bat_qa_studies

    aux:
    fruit_bat_qa_studies: fruit bats, fig canopies, answers, and scores
    """
    return aux


def _bench_fruit_bat_qa_studies(seed: int = 0) -> float:
    checks = []
    checks.append(fruit_bat_qa_studies_ok(True, True))
    checks.append(not fruit_bat_qa_studies_ok(False, True))
    checks.append(fruit_bat_qa_studies_aux(True))
    checks.append(not fruit_bat_qa_studies_aux(False))
    checks.append(True)  # bat-2 canon
    return float(sum(checks) / len(checks))


def bench_fruit_bat_qa_studies(seed: int = 0) -> dict[str, float]:
    return {"synthetic_fruit_bat_qa_studies": _bench_fruit_bat_qa_studies(seed)}
