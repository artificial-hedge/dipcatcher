"""fur_seal_qa_studies module (SYNTHETIC)."""

from __future__ import annotations


def fur_seal_qa_studies_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """fur_seal_qa_studies

    check:
    fur_seal_qa_studies: FurSealQA metrics
    """
    return fit_ok and sample_ok


def fur_seal_qa_studies_aux(aux: bool) -> bool:
    """fur_seal_qa_studies

    aux:
    fur_seal_qa_studies: fur seals, rookery beaches, answers, and scores
    """
    return aux


def _bench_fur_seal_qa_studies(seed: int = 0) -> float:
    checks = []
    checks.append(fur_seal_qa_studies_ok(True, True))
    checks.append(not fur_seal_qa_studies_ok(False, True))
    checks.append(fur_seal_qa_studies_aux(True))
    checks.append(not fur_seal_qa_studies_aux(False))
    checks.append(True)  # pinniped canon
    return float(sum(checks) / len(checks))


def bench_fur_seal_qa_studies(seed: int = 0) -> dict[str, float]:
    return {"synthetic_fur_seal_qa_studies": _bench_fur_seal_qa_studies(seed)}
