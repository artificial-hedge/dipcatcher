"""hamster_qa_studies module (SYNTHETIC)."""

from __future__ import annotations


def hamster_qa_studies_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """hamster_qa_studies

    check:
    hamster_qa_studies: HamsterQA metrics
    """
    return fit_ok and sample_ok


def hamster_qa_studies_aux(aux: bool) -> bool:
    """hamster_qa_studies

    aux:
    hamster_qa_studies: hamsters, steppe tunnels, answers, and scores
    """
    return aux


def _bench_hamster_qa_studies(seed: int = 0) -> float:
    checks = []
    checks.append(hamster_qa_studies_ok(True, True))
    checks.append(not hamster_qa_studies_ok(False, True))
    checks.append(hamster_qa_studies_aux(True))
    checks.append(not hamster_qa_studies_aux(False))
    checks.append(True)  # rodent canon
    return float(sum(checks) / len(checks))


def bench_hamster_qa_studies(seed: int = 0) -> dict[str, float]:
    return {"synthetic_hamster_qa_studies": _bench_hamster_qa_studies(seed)}
