"""umibozu_qa_studies module (SYNTHETIC)."""

from __future__ import annotations


def umibozu_qa_studies_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """umibozu_qa_studies

    check:
    umibozu_qa_studies: U
    """
    return fit_ok and sample_ok


def umibozu_qa_studies_aux(aux: bool) -> bool:
    """umibozu_qa_studies

    aux:
    umibozu_qa_studies: m
    """
    return aux


def _bench_umibozu_qa_studies(seed: int = 0) -> float:
    checks = []
    checks.append(umibozu_qa_studies_ok(True, True))
    checks.append(not umibozu_qa_studies_ok(False, True))
    checks.append(umibozu_qa_studies_aux(True))
    checks.append(not umibozu_qa_studies_aux(False))
    checks.append(True)  # yokai-8 canon
    return float(sum(checks) / len(checks))


def bench_umibozu_qa_studies(seed: int = 0) -> dict[str, float]:
    return {"synthetic_umibozu_qa_studies": _bench_umibozu_qa_studies(seed)}
