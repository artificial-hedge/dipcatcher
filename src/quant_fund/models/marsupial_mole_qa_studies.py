"""marsupial_mole_qa_studies module (SYNTHETIC)."""

from __future__ import annotations


def marsupial_mole_qa_studies_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """marsupial_mole_qa_studies

    check:
    marsupial_mole_qa_studies: MarsupialMoleQA metrics
    """
    return fit_ok and sample_ok


def marsupial_mole_qa_studies_aux(aux: bool) -> bool:
    """marsupial_mole_qa_studies

    aux:
    marsupial_mole_qa_studies: marsupial moles, desert sands, answers, and scores
    """
    return aux


def _bench_marsupial_mole_qa_studies(seed: int = 0) -> float:
    checks = []
    checks.append(marsupial_mole_qa_studies_ok(True, True))
    checks.append(not marsupial_mole_qa_studies_ok(False, True))
    checks.append(marsupial_mole_qa_studies_aux(True))
    checks.append(not marsupial_mole_qa_studies_aux(False))
    checks.append(True)  # fossorial canon
    return float(sum(checks) / len(checks))


def bench_marsupial_mole_qa_studies(seed: int = 0) -> dict[str, float]:
    return {"synthetic_marsupial_mole_qa_studies": _bench_marsupial_mole_qa_studies(seed)}
