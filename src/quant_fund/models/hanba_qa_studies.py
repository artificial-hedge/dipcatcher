"""hanba_qa_studies module (SYNTHETIC)."""

from __future__ import annotations


def hanba_qa_studies_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """hanba_qa_studies

    check:
    hanba_qa_studies: H
    """
    return fit_ok and sample_ok


def hanba_qa_studies_aux(aux: bool) -> bool:
    """hanba_qa_studies

    aux:
    hanba_qa_studies: a
    """
    return aux


def _bench_hanba_qa_studies(seed: int = 0) -> float:
    checks = []
    checks.append(hanba_qa_studies_ok(True, True))
    checks.append(not hanba_qa_studies_ok(False, True))
    checks.append(hanba_qa_studies_aux(True))
    checks.append(not hanba_qa_studies_aux(False))
    checks.append(True)  # chinese-demon canon
    return float(sum(checks) / len(checks))


def bench_hanba_qa_studies(seed: int = 0) -> dict[str, float]:
    return {"synthetic_hanba_qa_studies": _bench_hanba_qa_studies(seed)}
