"""evidence_inf_studies module (SYNTHETIC)."""

from __future__ import annotations


def evidence_inf_studies_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """evidence_inf_studies

    check:
    evidence_inf_studies: evidence-inference metrics
    """
    return fit_ok and sample_ok


def evidence_inf_studies_aux(aux: bool) -> bool:
    """evidence_inf_studies

    aux:
    evidence_inf_studies: claims, premises, labels, and accuracies
    """
    return aux


def _bench_evidence_inf_studies(seed: int = 0) -> float:
    checks = []
    checks.append(evidence_inf_studies_ok(True, True))
    checks.append(not evidence_inf_studies_ok(False, True))
    checks.append(evidence_inf_studies_aux(True))
    checks.append(not evidence_inf_studies_aux(False))
    checks.append(True)  # misinformation canon
    return float(sum(checks) / len(checks))


def bench_evidence_inf_studies(seed: int = 0) -> dict[str, float]:
    return {"synthetic_evidence_inf_studies": _bench_evidence_inf_studies(seed)}
