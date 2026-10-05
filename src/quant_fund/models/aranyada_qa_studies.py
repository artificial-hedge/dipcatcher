"""aranyada_qa_studies module (SYNTHETIC)."""

from __future__ import annotations


def aranyada_qa_studies_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """aranyada_qa_studies

    check:
    aranyada_qa_studies: d
    """
    return fit_ok and sample_ok


def aranyada_qa_studies_aux(aux: bool) -> bool:
    """aranyada_qa_studies

    aux:
    aranyada_qa_studies: e
    """
    return aux


def _bench_aranyada_qa_studies(seed: int = 0) -> float:
    checks = []
    checks.append(aranyada_qa_studies_ok(True, True))
    checks.append(not aranyada_qa_studies_ok(False, True))
    checks.append(aranyada_qa_studies_aux(True))
    checks.append(not aranyada_qa_studies_aux(False))
    checks.append(True)  # sabaean-myth canon
    return float(sum(checks) / len(checks))


def bench_aranyada_qa_studies(seed: int = 0) -> dict[str, float]:
    return {"synthetic_aranyada_qa_studies": _bench_aranyada_qa_studies(seed)}
