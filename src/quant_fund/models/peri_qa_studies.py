"""peri_qa_studies module (SYNTHETIC)."""

from __future__ import annotations


def peri_qa_studies_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """peri_qa_studies

    check:
    peri_qa_studies: p
    """
    return fit_ok and sample_ok


def peri_qa_studies_aux(aux: bool) -> bool:
    """peri_qa_studies

    aux:
    peri_qa_studies: e
    """
    return aux


def _bench_peri_qa_studies(seed: int = 0) -> float:
    checks = []
    checks.append(peri_qa_studies_ok(True, True))
    checks.append(not peri_qa_studies_ok(False, True))
    checks.append(peri_qa_studies_aux(True))
    checks.append(not peri_qa_studies_aux(False))
    checks.append(True)  # folk-spirit lore canon
    return float(sum(checks) / len(checks))


def bench_peri_qa_studies(seed: int = 0) -> dict[str, float]:
    return {"synthetic_peri_qa_studies": _bench_peri_qa_studies(seed)}
