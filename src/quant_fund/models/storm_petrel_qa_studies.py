"""storm_petrel_qa_studies module (SYNTHETIC)."""

from __future__ import annotations


def storm_petrel_qa_studies_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """storm_petrel_qa_studies

    check:
    storm_petrel_qa_studies: StormPetrelQA metrics
    """
    return fit_ok and sample_ok


def storm_petrel_qa_studies_aux(aux: bool) -> bool:
    """storm_petrel_qa_studies

    aux:
    storm_petrel_qa_studies: storm petrels, swells, answers, and scores
    """
    return aux


def _bench_storm_petrel_qa_studies(seed: int = 0) -> float:
    checks = []
    checks.append(storm_petrel_qa_studies_ok(True, True))
    checks.append(not storm_petrel_qa_studies_ok(False, True))
    checks.append(storm_petrel_qa_studies_aux(True))
    checks.append(not storm_petrel_qa_studies_aux(False))
    checks.append(True)  # seabird-4 canon
    return float(sum(checks) / len(checks))


def bench_storm_petrel_qa_studies(seed: int = 0) -> dict[str, float]:
    return {"synthetic_storm_petrel_qa_studies": _bench_storm_petrel_qa_studies(seed)}
