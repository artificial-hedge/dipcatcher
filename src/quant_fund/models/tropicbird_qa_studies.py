"""tropicbird_qa_studies module (SYNTHETIC)."""

from __future__ import annotations


def tropicbird_qa_studies_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """tropicbird_qa_studies

    check:
    tropicbird_qa_studies: TropicbirdQA metrics
    """
    return fit_ok and sample_ok


def tropicbird_qa_studies_aux(aux: bool) -> bool:
    """tropicbird_qa_studies

    aux:
    tropicbird_qa_studies: tropicbirds, thermals, answers, and scores
    """
    return aux


def _bench_tropicbird_qa_studies(seed: int = 0) -> float:
    checks = []
    checks.append(tropicbird_qa_studies_ok(True, True))
    checks.append(not tropicbird_qa_studies_ok(False, True))
    checks.append(tropicbird_qa_studies_aux(True))
    checks.append(not tropicbird_qa_studies_aux(False))
    checks.append(True)  # seabird-2 canon
    return float(sum(checks) / len(checks))


def bench_tropicbird_qa_studies(seed: int = 0) -> dict[str, float]:
    return {"synthetic_tropicbird_qa_studies": _bench_tropicbird_qa_studies(seed)}
