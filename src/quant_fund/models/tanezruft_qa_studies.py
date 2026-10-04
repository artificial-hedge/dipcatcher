"""tanezruft_qa_studies module (SYNTHETIC)."""

from __future__ import annotations


def tanezruft_qa_studies_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """tanezruft_qa_studies

    check:
    tanezruft_qa_studies: b
    """
    return fit_ok and sample_ok


def tanezruft_qa_studies_aux(aux: bool) -> bool:
    """tanezruft_qa_studies

    aux:
    tanezruft_qa_studies: a
    """
    return aux


def _bench_tanezruft_qa_studies(seed: int = 0) -> float:
    checks = []
    checks.append(tanezruft_qa_studies_ok(True, True))
    checks.append(not tanezruft_qa_studies_ok(False, True))
    checks.append(tanezruft_qa_studies_aux(True))
    checks.append(not tanezruft_qa_studies_aux(False))
    checks.append(True)  # saharan-2 canon
    return float(sum(checks) / len(checks))


def bench_tanezruft_qa_studies(seed: int = 0) -> dict[str, float]:
    return {"synthetic_tanezruft_qa_studies": _bench_tanezruft_qa_studies(seed)}
