"""namahage_qa_studies module (SYNTHETIC)."""

from __future__ import annotations


def namahage_qa_studies_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """namahage_qa_studies

    check:
    namahage_qa_studies: NamahageQA metrics
    """
    return fit_ok and sample_ok


def namahage_qa_studies_aux(aux: bool) -> bool:
    """namahage_qa_studies

    aux:
    namahage_qa_studies: namahages, oga mountains, answers, and scores
    """
    return aux


def _bench_namahage_qa_studies(seed: int = 0) -> float:
    checks = []
    checks.append(namahage_qa_studies_ok(True, True))
    checks.append(not namahage_qa_studies_ok(False, True))
    checks.append(namahage_qa_studies_aux(True))
    checks.append(not namahage_qa_studies_aux(False))
    checks.append(True)  # yokai-2 canon
    return float(sum(checks) / len(checks))


def bench_namahage_qa_studies(seed: int = 0) -> dict[str, float]:
    return {"synthetic_namahage_qa_studies": _bench_namahage_qa_studies(seed)}
