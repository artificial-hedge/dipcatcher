"""kuntilanak_qa_studies module (SYNTHETIC)."""

from __future__ import annotations


def kuntilanak_qa_studies_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """kuntilanak_qa_studies

    check:
    kuntilanak_qa_studies: K
    """
    return fit_ok and sample_ok


def kuntilanak_qa_studies_aux(aux: bool) -> bool:
    """kuntilanak_qa_studies

    aux:
    kuntilanak_qa_studies: u
    """
    return aux


def _bench_kuntilanak_qa_studies(seed: int = 0) -> float:
    checks = []
    checks.append(kuntilanak_qa_studies_ok(True, True))
    checks.append(not kuntilanak_qa_studies_ok(False, True))
    checks.append(kuntilanak_qa_studies_aux(True))
    checks.append(not kuntilanak_qa_studies_aux(False))
    checks.append(True)  # javanese-demon canon
    return float(sum(checks) / len(checks))


def bench_kuntilanak_qa_studies(seed: int = 0) -> dict[str, float]:
    return {"synthetic_kuntilanak_qa_studies": _bench_kuntilanak_qa_studies(seed)}
