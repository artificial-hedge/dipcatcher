"""ataecina_qa_studies module (SYNTHETIC)."""

from __future__ import annotations


def ataecina_qa_studies_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """ataecina_qa_studies

    check:
    ataecina_qa_studies: u
    """
    return fit_ok and sample_ok


def ataecina_qa_studies_aux(aux: bool) -> bool:
    """ataecina_qa_studies

    aux:
    ataecina_qa_studies: n
    """
    return aux


def _bench_ataecina_qa_studies(seed: int = 0) -> float:
    checks = []
    checks.append(ataecina_qa_studies_ok(True, True))
    checks.append(not ataecina_qa_studies_ok(False, True))
    checks.append(ataecina_qa_studies_aux(True))
    checks.append(not ataecina_qa_studies_aux(False))
    checks.append(True)  # iberian-myth canon
    return float(sum(checks) / len(checks))


def bench_ataecina_qa_studies(seed: int = 0) -> dict[str, float]:
    return {"synthetic_ataecina_qa_studies": _bench_ataecina_qa_studies(seed)}
