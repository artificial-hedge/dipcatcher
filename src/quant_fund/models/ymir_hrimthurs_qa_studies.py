"""ymir_hrimthurs_qa_studies module (SYNTHETIC)."""

from __future__ import annotations


def ymir_hrimthurs_qa_studies_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """ymir_hrimthurs_qa_studies

    check:
    ymir_hrimthurs_qa_studies: y
    """
    return fit_ok and sample_ok


def ymir_hrimthurs_qa_studies_aux(aux: bool) -> bool:
    """ymir_hrimthurs_qa_studies

    aux:
    ymir_hrimthurs_qa_studies: m
    """
    return aux


def _bench_ymir_hrimthurs_qa_studies(seed: int = 0) -> float:
    checks = []
    checks.append(ymir_hrimthurs_qa_studies_ok(True, True))
    checks.append(not ymir_hrimthurs_qa_studies_ok(False, True))
    checks.append(ymir_hrimthurs_qa_studies_aux(True))
    checks.append(not ymir_hrimthurs_qa_studies_aux(False))
    checks.append(True)  # folk-spirit lore canon
    return float(sum(checks) / len(checks))


def bench_ymir_hrimthurs_qa_studies(seed: int = 0) -> dict[str, float]:
    return {"synthetic_ymir_hrimthurs_qa_studies": _bench_ymir_hrimthurs_qa_studies(seed)}
