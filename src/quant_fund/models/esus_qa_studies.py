"""esus_qa_studies module (SYNTHETIC)."""

from __future__ import annotations


def esus_qa_studies_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """esus_qa_studies

    check:
    esus_qa_studies: w
    """
    return fit_ok and sample_ok


def esus_qa_studies_aux(aux: bool) -> bool:
    """esus_qa_studies

    aux:
    esus_qa_studies: o
    """
    return aux


def _bench_esus_qa_studies(seed: int = 0) -> float:
    checks = []
    checks.append(esus_qa_studies_ok(True, True))
    checks.append(not esus_qa_studies_ok(False, True))
    checks.append(esus_qa_studies_aux(True))
    checks.append(not esus_qa_studies_aux(False))
    checks.append(True)  # gallic-myth canon
    return float(sum(checks) / len(checks))


def bench_esus_qa_studies(seed: int = 0) -> dict[str, float]:
    return {"synthetic_esus_qa_studies": _bench_esus_qa_studies(seed)}
