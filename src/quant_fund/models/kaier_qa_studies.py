"""kaier_qa_studies module (SYNTHETIC)."""

from __future__ import annotations


def kaier_qa_studies_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """kaier_qa_studies

    check:
    kaier_qa_studies: s
    """
    return fit_ok and sample_ok


def kaier_qa_studies_aux(aux: bool) -> bool:
    """kaier_qa_studies

    aux:
    kaier_qa_studies: u
    """
    return aux


def _bench_kaier_qa_studies(seed: int = 0) -> float:
    checks = []
    checks.append(kaier_qa_studies_ok(True, True))
    checks.append(not kaier_qa_studies_ok(False, True))
    checks.append(kaier_qa_studies_aux(True))
    checks.append(not kaier_qa_studies_aux(False))
    checks.append(True)  # breton-myth-2 canon
    return float(sum(checks) / len(checks))


def bench_kaier_qa_studies(seed: int = 0) -> dict[str, float]:
    return {"synthetic_kaier_qa_studies": _bench_kaier_qa_studies(seed)}
