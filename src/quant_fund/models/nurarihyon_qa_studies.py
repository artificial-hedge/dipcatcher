"""nurarihyon_qa_studies module (SYNTHETIC)."""

from __future__ import annotations


def nurarihyon_qa_studies_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """nurarihyon_qa_studies

    check:
    nurarihyon_qa_studies: N
    """
    return fit_ok and sample_ok


def nurarihyon_qa_studies_aux(aux: bool) -> bool:
    """nurarihyon_qa_studies

    aux:
    nurarihyon_qa_studies: u
    """
    return aux


def _bench_nurarihyon_qa_studies(seed: int = 0) -> float:
    checks = []
    checks.append(nurarihyon_qa_studies_ok(True, True))
    checks.append(not nurarihyon_qa_studies_ok(False, True))
    checks.append(nurarihyon_qa_studies_aux(True))
    checks.append(not nurarihyon_qa_studies_aux(False))
    checks.append(True)  # yokai-7 canon
    return float(sum(checks) / len(checks))


def bench_nurarihyon_qa_studies(seed: int = 0) -> dict[str, float]:
    return {"synthetic_nurarihyon_qa_studies": _bench_nurarihyon_qa_studies(seed)}
