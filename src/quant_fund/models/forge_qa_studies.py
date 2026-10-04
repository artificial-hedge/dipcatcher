"""forge_qa_studies module (SYNTHETIC)."""

from __future__ import annotations


def forge_qa_studies_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """forge_qa_studies

    check:
    forge_qa_studies: ForgeQA metrics
    """
    return fit_ok and sample_ok


def forge_qa_studies_aux(aux: bool) -> bool:
    """forge_qa_studies

    aux:
    forge_qa_studies: forges, anvils, answers, and scores
    """
    return aux


def _bench_forge_qa_studies(seed: int = 0) -> float:
    checks = []
    checks.append(forge_qa_studies_ok(True, True))
    checks.append(not forge_qa_studies_ok(False, True))
    checks.append(forge_qa_studies_aux(True))
    checks.append(not forge_qa_studies_aux(False))
    checks.append(True)  # forge canon
    return float(sum(checks) / len(checks))


def bench_forge_qa_studies(seed: int = 0) -> dict[str, float]:
    return {"synthetic_forge_qa_studies": _bench_forge_qa_studies(seed)}
