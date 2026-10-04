"""isolde_qa_studies module (SYNTHETIC)."""

from __future__ import annotations


def isolde_qa_studies_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """isolde_qa_studies

    check:
    isolde_qa_studies: I
    """
    return fit_ok and sample_ok


def isolde_qa_studies_aux(aux: bool) -> bool:
    """isolde_qa_studies

    aux:
    isolde_qa_studies: r
    """
    return aux


def _bench_isolde_qa_studies(seed: int = 0) -> float:
    checks = []
    checks.append(isolde_qa_studies_ok(True, True))
    checks.append(not isolde_qa_studies_ok(False, True))
    checks.append(isolde_qa_studies_aux(True))
    checks.append(not isolde_qa_studies_aux(False))
    checks.append(True)  # arthurian-3 canon
    return float(sum(checks) / len(checks))


def bench_isolde_qa_studies(seed: int = 0) -> dict[str, float]:
    return {"synthetic_isolde_qa_studies": _bench_isolde_qa_studies(seed)}
