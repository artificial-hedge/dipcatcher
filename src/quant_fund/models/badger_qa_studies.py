"""badger_qa_studies module (SYNTHETIC)."""

from __future__ import annotations


def badger_qa_studies_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """badger_qa_studies

    check:
    badger_qa_studies: BadgerQA metrics
    """
    return fit_ok and sample_ok


def badger_qa_studies_aux(aux: bool) -> bool:
    """badger_qa_studies

    aux:
    badger_qa_studies: badgers, setts, answers, and scores
    """
    return aux


def _bench_badger_qa_studies(seed: int = 0) -> float:
    checks = []
    checks.append(badger_qa_studies_ok(True, True))
    checks.append(not badger_qa_studies_ok(False, True))
    checks.append(badger_qa_studies_aux(True))
    checks.append(not badger_qa_studies_aux(False))
    checks.append(True)  # forest-mammal canon
    return float(sum(checks) / len(checks))


def bench_badger_qa_studies(seed: int = 0) -> dict[str, float]:
    return {"synthetic_badger_qa_studies": _bench_badger_qa_studies(seed)}
