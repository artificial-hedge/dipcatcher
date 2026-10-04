"""nurikabe_qa_studies module (SYNTHETIC)."""

from __future__ import annotations


def nurikabe_qa_studies_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """nurikabe_qa_studies

    check:
    nurikabe_qa_studies: NurikabeQA metrics
    """
    return fit_ok and sample_ok


def nurikabe_qa_studies_aux(aux: bool) -> bool:
    """nurikabe_qa_studies

    aux:
    nurikabe_qa_studies: nurikabes, night roads, answers, and scores
    """
    return aux


def _bench_nurikabe_qa_studies(seed: int = 0) -> float:
    checks = []
    checks.append(nurikabe_qa_studies_ok(True, True))
    checks.append(not nurikabe_qa_studies_ok(False, True))
    checks.append(nurikabe_qa_studies_aux(True))
    checks.append(not nurikabe_qa_studies_aux(False))
    checks.append(True)  # yokai-4 canon
    return float(sum(checks) / len(checks))


def bench_nurikabe_qa_studies(seed: int = 0) -> dict[str, float]:
    return {"synthetic_nurikabe_qa_studies": _bench_nurikabe_qa_studies(seed)}
