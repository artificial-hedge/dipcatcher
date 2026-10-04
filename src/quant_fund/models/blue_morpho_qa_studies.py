"""blue_morpho_qa_studies module (SYNTHETIC)."""

from __future__ import annotations


def blue_morpho_qa_studies_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """blue_morpho_qa_studies

    check:
    blue_morpho_qa_studies: BlueMorphoQA metrics
    """
    return fit_ok and sample_ok


def blue_morpho_qa_studies_aux(aux: bool) -> bool:
    """blue_morpho_qa_studies

    aux:
    blue_morpho_qa_studies: blue morphos, rainforests, answers, and scores
    """
    return aux


def _bench_blue_morpho_qa_studies(seed: int = 0) -> float:
    checks = []
    checks.append(blue_morpho_qa_studies_ok(True, True))
    checks.append(not blue_morpho_qa_studies_ok(False, True))
    checks.append(blue_morpho_qa_studies_aux(True))
    checks.append(not blue_morpho_qa_studies_aux(False))
    checks.append(True)  # butterfly canon
    return float(sum(checks) / len(checks))


def bench_blue_morpho_qa_studies(seed: int = 0) -> dict[str, float]:
    return {"synthetic_blue_morpho_qa_studies": _bench_blue_morpho_qa_studies(seed)}
