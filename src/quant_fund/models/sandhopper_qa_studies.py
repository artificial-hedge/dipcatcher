"""sandhopper_qa_studies module (SYNTHETIC)."""

from __future__ import annotations


def sandhopper_qa_studies_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """sandhopper_qa_studies

    check:
    sandhopper_qa_studies: SandhopperQA metrics
    """
    return fit_ok and sample_ok


def sandhopper_qa_studies_aux(aux: bool) -> bool:
    """sandhopper_qa_studies

    aux:
    sandhopper_qa_studies: sandhoppers, dune wrack, answers, and scores
    """
    return aux


def _bench_sandhopper_qa_studies(seed: int = 0) -> float:
    checks = []
    checks.append(sandhopper_qa_studies_ok(True, True))
    checks.append(not sandhopper_qa_studies_ok(False, True))
    checks.append(sandhopper_qa_studies_aux(True))
    checks.append(not sandhopper_qa_studies_aux(False))
    checks.append(True)  # plankton-shore canon
    return float(sum(checks) / len(checks))


def bench_sandhopper_qa_studies(seed: int = 0) -> dict[str, float]:
    return {"synthetic_sandhopper_qa_studies": _bench_sandhopper_qa_studies(seed)}
