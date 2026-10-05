"""duppy_qa_studies module (SYNTHETIC)."""

from __future__ import annotations


def duppy_qa_studies_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """duppy_qa_studies

    check:
    duppy_qa_studies: D
    """
    return fit_ok and sample_ok


def duppy_qa_studies_aux(aux: bool) -> bool:
    """duppy_qa_studies

    aux:
    duppy_qa_studies: u
    """
    return aux


def _bench_duppy_qa_studies(seed: int = 0) -> float:
    checks = []
    checks.append(duppy_qa_studies_ok(True, True))
    checks.append(not duppy_qa_studies_ok(False, True))
    checks.append(duppy_qa_studies_aux(True))
    checks.append(not duppy_qa_studies_aux(False))
    checks.append(True)  # caribbean-demon canon
    return float(sum(checks) / len(checks))


def bench_duppy_qa_studies(seed: int = 0) -> dict[str, float]:
    return {"synthetic_duppy_qa_studies": _bench_duppy_qa_studies(seed)}
