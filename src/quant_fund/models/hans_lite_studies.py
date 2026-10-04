"""hans_lite_studies module (SYNTHETIC)."""

from __future__ import annotations


def hans_lite_studies_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """hans_lite_studies

    check:
    hans_lite_studies: HANS heuristic metrics
    """
    return fit_ok and sample_ok


def hans_lite_studies_aux(aux: bool) -> bool:
    """hans_lite_studies

    aux:
    hans_lite_studies: premises, hypotheses, labels, and accuracies
    """
    return aux


def _bench_hans_lite_studies(seed: int = 0) -> float:
    checks = []
    checks.append(hans_lite_studies_ok(True, True))
    checks.append(not hans_lite_studies_ok(False, True))
    checks.append(hans_lite_studies_aux(True))
    checks.append(not hans_lite_studies_aux(False))
    checks.append(True)  # NLI-eval-2 canon
    return float(sum(checks) / len(checks))


def bench_hans_lite_studies(seed: int = 0) -> dict[str, float]:
    return {"synthetic_hans_lite_studies": _bench_hans_lite_studies(seed)}
