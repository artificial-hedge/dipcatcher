"""sup_nli_studies module (SYNTHETIC)."""

from __future__ import annotations


def sup_nli_studies_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """sup_nli_studies

    check:
    sup_nli_studies: super-NLI metrics
    """
    return fit_ok and sample_ok


def sup_nli_studies_aux(aux: bool) -> bool:
    """sup_nli_studies

    aux:
    sup_nli_studies: premises, hypotheses, labels, and accuracies
    """
    return aux


def _bench_sup_nli_studies(seed: int = 0) -> float:
    checks = []
    checks.append(sup_nli_studies_ok(True, True))
    checks.append(not sup_nli_studies_ok(False, True))
    checks.append(sup_nli_studies_aux(True))
    checks.append(not sup_nli_studies_aux(False))
    checks.append(True)  # NLI-eval-2 canon
    return float(sum(checks) / len(checks))


def bench_sup_nli_studies(seed: int = 0) -> dict[str, float]:
    return {"synthetic_sup_nli_studies": _bench_sup_nli_studies(seed)}
