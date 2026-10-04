"""arch_qa_studies module (SYNTHETIC)."""

from __future__ import annotations


def arch_qa_studies_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """arch_qa_studies

    check:
    arch_qa_studies: ArchQA metrics
    """
    return fit_ok and sample_ok


def arch_qa_studies_aux(aux: bool) -> bool:
    """arch_qa_studies

    aux:
    arch_qa_studies: arches, spans, answers, and scores
    """
    return aux


def _bench_arch_qa_studies(seed: int = 0) -> float:
    checks = []
    checks.append(arch_qa_studies_ok(True, True))
    checks.append(not arch_qa_studies_ok(False, True))
    checks.append(arch_qa_studies_aux(True))
    checks.append(not arch_qa_studies_aux(False))
    checks.append(True)  # highland canon
    return float(sum(checks) / len(checks))


def bench_arch_qa_studies(seed: int = 0) -> dict[str, float]:
    return {"synthetic_arch_qa_studies": _bench_arch_qa_studies(seed)}
