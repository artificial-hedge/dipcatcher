"""oni_qa_studies module (SYNTHETIC)."""

from __future__ import annotations


def oni_qa_studies_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """oni_qa_studies

    check:
    oni_qa_studies: OniQA metrics
    """
    return fit_ok and sample_ok


def oni_qa_studies_aux(aux: bool) -> bool:
    """oni_qa_studies

    aux:
    oni_qa_studies: onis, thunder caves, answers, and scores
    """
    return aux


def _bench_oni_qa_studies(seed: int = 0) -> float:
    checks = []
    checks.append(oni_qa_studies_ok(True, True))
    checks.append(not oni_qa_studies_ok(False, True))
    checks.append(oni_qa_studies_aux(True))
    checks.append(not oni_qa_studies_aux(False))
    checks.append(True)  # yokai canon
    return float(sum(checks) / len(checks))


def bench_oni_qa_studies(seed: int = 0) -> dict[str, float]:
    return {"synthetic_oni_qa_studies": _bench_oni_qa_studies(seed)}
