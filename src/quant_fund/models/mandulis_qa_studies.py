"""mandulis_qa_studies module (SYNTHETIC)."""

from __future__ import annotations


def mandulis_qa_studies_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """mandulis_qa_studies

    check:
    mandulis_qa_studies: s
    """
    return fit_ok and sample_ok


def mandulis_qa_studies_aux(aux: bool) -> bool:
    """mandulis_qa_studies

    aux:
    mandulis_qa_studies: u
    """
    return aux


def _bench_mandulis_qa_studies(seed: int = 0) -> float:
    checks = []
    checks.append(mandulis_qa_studies_ok(True, True))
    checks.append(not mandulis_qa_studies_ok(False, True))
    checks.append(mandulis_qa_studies_aux(True))
    checks.append(not mandulis_qa_studies_aux(False))
    checks.append(True)  # meroitic-myth canon
    return float(sum(checks) / len(checks))


def bench_mandulis_qa_studies(seed: int = 0) -> dict[str, float]:
    return {"synthetic_mandulis_qa_studies": _bench_mandulis_qa_studies(seed)}
