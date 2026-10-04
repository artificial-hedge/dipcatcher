"""kishi_demon_qa_studies module (SYNTHETIC)."""

from __future__ import annotations


def kishi_demon_qa_studies_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """kishi_demon_qa_studies

    check:
    kishi_demon_qa_studies: K
    """
    return fit_ok and sample_ok


def kishi_demon_qa_studies_aux(aux: bool) -> bool:
    """kishi_demon_qa_studies

    aux:
    kishi_demon_qa_studies: i
    """
    return aux


def _bench_kishi_demon_qa_studies(seed: int = 0) -> float:
    checks = []
    checks.append(kishi_demon_qa_studies_ok(True, True))
    checks.append(not kishi_demon_qa_studies_ok(False, True))
    checks.append(kishi_demon_qa_studies_aux(True))
    checks.append(not kishi_demon_qa_studies_aux(False))
    checks.append(True)  # african-demon canon
    return float(sum(checks) / len(checks))


def bench_kishi_demon_qa_studies(seed: int = 0) -> dict[str, float]:
    return {"synthetic_kishi_demon_qa_studies": _bench_kishi_demon_qa_studies(seed)}
