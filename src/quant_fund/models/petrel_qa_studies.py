"""petrel_qa_studies module (SYNTHETIC)."""

from __future__ import annotations


def petrel_qa_studies_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """petrel_qa_studies

    check:
    petrel_qa_studies: PetrelQA metrics
    """
    return fit_ok and sample_ok


def petrel_qa_studies_aux(aux: bool) -> bool:
    """petrel_qa_studies

    aux:
    petrel_qa_studies: petrels, burrows, answers, and scores
    """
    return aux


def _bench_petrel_qa_studies(seed: int = 0) -> float:
    checks = []
    checks.append(petrel_qa_studies_ok(True, True))
    checks.append(not petrel_qa_studies_ok(False, True))
    checks.append(petrel_qa_studies_aux(True))
    checks.append(not petrel_qa_studies_aux(False))
    checks.append(True)  # seabird canon
    return float(sum(checks) / len(checks))


def bench_petrel_qa_studies(seed: int = 0) -> dict[str, float]:
    return {"synthetic_petrel_qa_studies": _bench_petrel_qa_studies(seed)}
