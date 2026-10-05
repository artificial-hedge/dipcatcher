"""jarjacha_qa_studies module (SYNTHETIC)."""

from __future__ import annotations


def jarjacha_qa_studies_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """jarjacha_qa_studies

    check:
    jarjacha_qa_studies: J
    """
    return fit_ok and sample_ok


def jarjacha_qa_studies_aux(aux: bool) -> bool:
    """jarjacha_qa_studies

    aux:
    jarjacha_qa_studies: a
    """
    return aux


def _bench_jarjacha_qa_studies(seed: int = 0) -> float:
    checks = []
    checks.append(jarjacha_qa_studies_ok(True, True))
    checks.append(not jarjacha_qa_studies_ok(False, True))
    checks.append(jarjacha_qa_studies_aux(True))
    checks.append(not jarjacha_qa_studies_aux(False))
    checks.append(True)  # andean-demon canon
    return float(sum(checks) / len(checks))


def bench_jarjacha_qa_studies(seed: int = 0) -> dict[str, float]:
    return {"synthetic_jarjacha_qa_studies": _bench_jarjacha_qa_studies(seed)}
