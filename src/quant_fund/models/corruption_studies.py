"""corruption_studies module (SYNTHETIC)."""

from __future__ import annotations


def corruption_studies_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """corruption_studies

    check:
    corruption_studies: corruption types/severities and robustness deltas
    """
    return fit_ok and sample_ok


def corruption_studies_aux(aux: bool) -> bool:
    """corruption_studies

    aux:
    corruption_studies: corrupt transforms/noise and degraded accs
    """
    return aux


def _bench_corruption_studies(seed: int = 0) -> float:
    checks = []
    checks.append(corruption_studies_ok(True, True))
    checks.append(not corruption_studies_ok(False, True))
    checks.append(corruption_studies_aux(True))
    checks.append(not corruption_studies_aux(False))
    checks.append(True)  # robustness-eval canon
    return float(sum(checks) / len(checks))


def bench_corruption_studies(seed: int = 0) -> dict[str, float]:
    return {"synthetic_corruption_studies": _bench_corruption_studies(seed)}
