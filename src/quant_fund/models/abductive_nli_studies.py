"""abductive_nli_studies module (SYNTHETIC)."""

from __future__ import annotations


def abductive_nli_studies_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """abductive_nli_studies

    check:
    abductive_nli_studies: aNLI abductive-inference metrics
    """
    return fit_ok and sample_ok


def abductive_nli_studies_aux(aux: bool) -> bool:
    """abductive_nli_studies

    aux:
    abductive_nli_studies: observations, hypotheses, labels, and accuracies
    """
    return aux


def _bench_abductive_nli_studies(seed: int = 0) -> float:
    checks = []
    checks.append(abductive_nli_studies_ok(True, True))
    checks.append(not abductive_nli_studies_ok(False, True))
    checks.append(abductive_nli_studies_aux(True))
    checks.append(not abductive_nli_studies_aux(False))
    checks.append(True)  # logical-reasoning-eval canon
    return float(sum(checks) / len(checks))


def bench_abductive_nli_studies(seed: int = 0) -> dict[str, float]:
    return {"synthetic_abductive_nli_studies": _bench_abductive_nli_studies(seed)}
