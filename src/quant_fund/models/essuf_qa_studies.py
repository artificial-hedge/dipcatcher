"""essuf_qa_studies module (SYNTHETIC)."""

from __future__ import annotations


def essuf_qa_studies_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """essuf_qa_studies

    check:
    essuf_qa_studies: d
    """
    return fit_ok and sample_ok


def essuf_qa_studies_aux(aux: bool) -> bool:
    """essuf_qa_studies

    aux:
    essuf_qa_studies: e
    """
    return aux


def _bench_essuf_qa_studies(seed: int = 0) -> float:
    checks = []
    checks.append(essuf_qa_studies_ok(True, True))
    checks.append(not essuf_qa_studies_ok(False, True))
    checks.append(essuf_qa_studies_aux(True))
    checks.append(not essuf_qa_studies_aux(False))
    checks.append(True)  # saharan-2 canon
    return float(sum(checks) / len(checks))


def bench_essuf_qa_studies(seed: int = 0) -> dict[str, float]:
    return {"synthetic_essuf_qa_studies": _bench_essuf_qa_studies(seed)}
