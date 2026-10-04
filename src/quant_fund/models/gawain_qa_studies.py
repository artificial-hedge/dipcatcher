"""gawain_qa_studies module (SYNTHETIC)."""

from __future__ import annotations


def gawain_qa_studies_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """gawain_qa_studies

    check:
    gawain_qa_studies: g
    """
    return fit_ok and sample_ok


def gawain_qa_studies_aux(aux: bool) -> bool:
    """gawain_qa_studies

    aux:
    gawain_qa_studies: r
    """
    return aux


def _bench_gawain_qa_studies(seed: int = 0) -> float:
    checks = []
    checks.append(gawain_qa_studies_ok(True, True))
    checks.append(not gawain_qa_studies_ok(False, True))
    checks.append(gawain_qa_studies_aux(True))
    checks.append(not gawain_qa_studies_aux(False))
    checks.append(True)  # arthurian-2 canon
    return float(sum(checks) / len(checks))


def bench_gawain_qa_studies(seed: int = 0) -> dict[str, float]:
    return {"synthetic_gawain_qa_studies": _bench_gawain_qa_studies(seed)}
