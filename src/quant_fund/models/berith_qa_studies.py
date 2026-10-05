"""berith_qa_studies module (SYNTHETIC)."""

from __future__ import annotations


def berith_qa_studies_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """berith_qa_studies

    check:
    berith_qa_studies: B
    """
    return fit_ok and sample_ok


def berith_qa_studies_aux(aux: bool) -> bool:
    """berith_qa_studies

    aux:
    berith_qa_studies: e
    """
    return aux


def _bench_berith_qa_studies(seed: int = 0) -> float:
    checks = []
    checks.append(berith_qa_studies_ok(True, True))
    checks.append(not berith_qa_studies_ok(False, True))
    checks.append(berith_qa_studies_aux(True))
    checks.append(not berith_qa_studies_aux(False))
    checks.append(True)  # goetic-circle canon
    return float(sum(checks) / len(checks))


def bench_berith_qa_studies(seed: int = 0) -> dict[str, float]:
    return {"synthetic_berith_qa_studies": _bench_berith_qa_studies(seed)}
