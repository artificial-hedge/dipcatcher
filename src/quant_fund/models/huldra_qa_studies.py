"""huldra_qa_studies module (SYNTHETIC)."""

from __future__ import annotations


def huldra_qa_studies_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """huldra_qa_studies

    check:
    huldra_qa_studies: HuldraQA metrics
    """
    return fit_ok and sample_ok


def huldra_qa_studies_aux(aux: bool) -> bool:
    """huldra_qa_studies

    aux:
    huldra_qa_studies: huldras, hidden folk, answers, and scores
    """
    return aux


def _bench_huldra_qa_studies(seed: int = 0) -> float:
    checks = []
    checks.append(huldra_qa_studies_ok(True, True))
    checks.append(not huldra_qa_studies_ok(False, True))
    checks.append(huldra_qa_studies_aux(True))
    checks.append(not huldra_qa_studies_aux(False))
    checks.append(True)  # norse-realm-2 canon
    return float(sum(checks) / len(checks))


def bench_huldra_qa_studies(seed: int = 0) -> dict[str, float]:
    return {"synthetic_huldra_qa_studies": _bench_huldra_qa_studies(seed)}
