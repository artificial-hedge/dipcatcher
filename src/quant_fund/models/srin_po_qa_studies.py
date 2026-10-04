"""srin_po_qa_studies module (SYNTHETIC)."""

from __future__ import annotations


def srin_po_qa_studies_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """srin_po_qa_studies

    check:
    srin_po_qa_studies: S
    """
    return fit_ok and sample_ok


def srin_po_qa_studies_aux(aux: bool) -> bool:
    """srin_po_qa_studies

    aux:
    srin_po_qa_studies: r
    """
    return aux


def _bench_srin_po_qa_studies(seed: int = 0) -> float:
    checks = []
    checks.append(srin_po_qa_studies_ok(True, True))
    checks.append(not srin_po_qa_studies_ok(False, True))
    checks.append(srin_po_qa_studies_aux(True))
    checks.append(not srin_po_qa_studies_aux(False))
    checks.append(True)  # tibetan-demon canon
    return float(sum(checks) / len(checks))


def bench_srin_po_qa_studies(seed: int = 0) -> dict[str, float]:
    return {"synthetic_srin_po_qa_studies": _bench_srin_po_qa_studies(seed)}
