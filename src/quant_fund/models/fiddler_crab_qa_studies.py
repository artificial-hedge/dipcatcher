"""fiddler_crab_qa_studies module (SYNTHETIC)."""

from __future__ import annotations


def fiddler_crab_qa_studies_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """fiddler_crab_qa_studies

    check:
    fiddler_crab_qa_studies: FiddlerCrabQA metrics
    """
    return fit_ok and sample_ok


def fiddler_crab_qa_studies_aux(aux: bool) -> bool:
    """fiddler_crab_qa_studies

    aux:
    fiddler_crab_qa_studies: fiddler crabs, tidal flats, answers, and scores
    """
    return aux


def _bench_fiddler_crab_qa_studies(seed: int = 0) -> float:
    checks = []
    checks.append(fiddler_crab_qa_studies_ok(True, True))
    checks.append(not fiddler_crab_qa_studies_ok(False, True))
    checks.append(fiddler_crab_qa_studies_aux(True))
    checks.append(not fiddler_crab_qa_studies_aux(False))
    checks.append(True)  # crab canon
    return float(sum(checks) / len(checks))


def bench_fiddler_crab_qa_studies(seed: int = 0) -> dict[str, float]:
    return {"synthetic_fiddler_crab_qa_studies": _bench_fiddler_crab_qa_studies(seed)}
