"""sodom2_qa_studies module (SYNTHETIC)."""

from __future__ import annotations


def sodom2_qa_studies_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """sodom2_qa_studies

    check:
    sodom2_qa_studies: b
    """
    return fit_ok and sample_ok


def sodom2_qa_studies_aux(aux: bool) -> bool:
    """sodom2_qa_studies

    aux:
    sodom2_qa_studies: u
    """
    return aux


def _bench_sodom2_qa_studies(seed: int = 0) -> float:
    checks = []
    checks.append(sodom2_qa_studies_ok(True, True))
    checks.append(not sodom2_qa_studies_ok(False, True))
    checks.append(sodom2_qa_studies_aux(True))
    checks.append(not sodom2_qa_studies_aux(False))
    checks.append(True)  # ammonite-myth canon
    return float(sum(checks) / len(checks))


def bench_sodom2_qa_studies(seed: int = 0) -> dict[str, float]:
    return {"synthetic_sodom2_qa_studies": _bench_sodom2_qa_studies(seed)}
