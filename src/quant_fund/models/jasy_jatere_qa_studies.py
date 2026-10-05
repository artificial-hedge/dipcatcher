"""jasy_jatere_qa_studies module (SYNTHETIC)."""

from __future__ import annotations


def jasy_jatere_qa_studies_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """jasy_jatere_qa_studies

    check:
    jasy_jatere_qa_studies: J
    """
    return fit_ok and sample_ok


def jasy_jatere_qa_studies_aux(aux: bool) -> bool:
    """jasy_jatere_qa_studies

    aux:
    jasy_jatere_qa_studies: a
    """
    return aux


def _bench_jasy_jatere_qa_studies(seed: int = 0) -> float:
    checks = []
    checks.append(jasy_jatere_qa_studies_ok(True, True))
    checks.append(not jasy_jatere_qa_studies_ok(False, True))
    checks.append(jasy_jatere_qa_studies_aux(True))
    checks.append(not jasy_jatere_qa_studies_aux(False))
    checks.append(True)  # guarani-demon canon
    return float(sum(checks) / len(checks))


def bench_jasy_jatere_qa_studies(seed: int = 0) -> dict[str, float]:
    return {"synthetic_jasy_jatere_qa_studies": _bench_jasy_jatere_qa_studies(seed)}
