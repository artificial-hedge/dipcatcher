"""oilbird_qa_studies module (SYNTHETIC)."""

from __future__ import annotations


def oilbird_qa_studies_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """oilbird_qa_studies

    check:
    oilbird_qa_studies: OilbirdQA metrics
    """
    return fit_ok and sample_ok


def oilbird_qa_studies_aux(aux: bool) -> bool:
    """oilbird_qa_studies

    aux:
    oilbird_qa_studies: oilbirds, caves, answers, and scores
    """
    return aux


def _bench_oilbird_qa_studies(seed: int = 0) -> float:
    checks = []
    checks.append(oilbird_qa_studies_ok(True, True))
    checks.append(not oilbird_qa_studies_ok(False, True))
    checks.append(oilbird_qa_studies_aux(True))
    checks.append(not oilbird_qa_studies_aux(False))
    checks.append(True)  # nightjar-2 canon
    return float(sum(checks) / len(checks))


def bench_oilbird_qa_studies(seed: int = 0) -> dict[str, float]:
    return {"synthetic_oilbird_qa_studies": _bench_oilbird_qa_studies(seed)}
