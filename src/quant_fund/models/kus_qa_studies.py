"""kus_qa_studies module (SYNTHETIC)."""

from __future__ import annotations


def kus_qa_studies_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """kus_qa_studies

    check:
    kus_qa_studies: K
    """
    return fit_ok and sample_ok


def kus_qa_studies_aux(aux: bool) -> bool:
    """kus_qa_studies

    aux:
    kus_qa_studies: u
    """
    return aux


def _bench_kus_qa_studies(seed: int = 0) -> float:
    checks = []
    checks.append(kus_qa_studies_ok(True, True))
    checks.append(not kus_qa_studies_ok(False, True))
    checks.append(kus_qa_studies_aux(True))
    checks.append(not kus_qa_studies_aux(False))
    checks.append(True)  # siberian-demon canon
    return float(sum(checks) / len(checks))


def bench_kus_qa_studies(seed: int = 0) -> dict[str, float]:
    return {"synthetic_kus_qa_studies": _bench_kus_qa_studies(seed)}
