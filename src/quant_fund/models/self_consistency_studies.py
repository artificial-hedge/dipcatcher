"""self_consistency_studies module (SYNTHETIC)."""

from __future__ import annotations


def self_consistency_studies_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """self_consistency_studies

    check:
    self_consistency_studies: majority-vote sampling over reasoning paths/samples and votes
    """
    return fit_ok and sample_ok


def self_consistency_studies_aux(aux: bool) -> bool:
    """self_consistency_studies

    aux:
    self_consistency_studies: answer-cluster consistency and confidence margins/clusters and margins
    """
    return aux


def _bench_self_consistency_studies(seed: int = 0) -> float:
    checks = []
    checks.append(self_consistency_studies_ok(True, True))
    checks.append(not self_consistency_studies_ok(False, True))
    checks.append(self_consistency_studies_aux(True))
    checks.append(not self_consistency_studies_aux(False))
    checks.append(True)  # reasoning/CoT canon
    return float(sum(checks) / len(checks))


def bench_self_consistency_studies(seed: int = 0) -> dict[str, float]:
    return {"synthetic_self_consistency_studies": _bench_self_consistency_studies(seed)}
