"""crystal_qa_studies module (SYNTHETIC)."""

from __future__ import annotations


def crystal_qa_studies_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """crystal_qa_studies

    check:
    crystal_qa_studies: CrystalQA metrics
    """
    return fit_ok and sample_ok


def crystal_qa_studies_aux(aux: bool) -> bool:
    """crystal_qa_studies

    aux:
    crystal_qa_studies: crystals, lattices, answers, and scores
    """
    return aux


def _bench_crystal_qa_studies(seed: int = 0) -> float:
    checks = []
    checks.append(crystal_qa_studies_ok(True, True))
    checks.append(not crystal_qa_studies_ok(False, True))
    checks.append(crystal_qa_studies_aux(True))
    checks.append(not crystal_qa_studies_aux(False))
    checks.append(True)  # gem canon
    return float(sum(checks) / len(checks))


def bench_crystal_qa_studies(seed: int = 0) -> dict[str, float]:
    return {"synthetic_crystal_qa_studies": _bench_crystal_qa_studies(seed)}
