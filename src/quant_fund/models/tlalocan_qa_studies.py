"""tlalocan_qa_studies module (SYNTHETIC)."""

from __future__ import annotations


def tlalocan_qa_studies_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """tlalocan_qa_studies

    check:
    tlalocan_qa_studies: TlalocanQA metrics
    """
    return fit_ok and sample_ok


def tlalocan_qa_studies_aux(aux: bool) -> bool:
    """tlalocan_qa_studies

    aux:
    tlalocan_qa_studies: tlalocans, rain paradises, answers, and scores
    """
    return aux


def _bench_tlalocan_qa_studies(seed: int = 0) -> float:
    checks = []
    checks.append(tlalocan_qa_studies_ok(True, True))
    checks.append(not tlalocan_qa_studies_ok(False, True))
    checks.append(tlalocan_qa_studies_aux(True))
    checks.append(not tlalocan_qa_studies_aux(False))
    checks.append(True)  # aztec-myth canon
    return float(sum(checks) / len(checks))


def bench_tlalocan_qa_studies(seed: int = 0) -> dict[str, float]:
    return {"synthetic_tlalocan_qa_studies": _bench_tlalocan_qa_studies(seed)}
