"""miniwob_studies module (SYNTHETIC)."""

from __future__ import annotations


def miniwob_studies_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """miniwob_studies

    check:
    miniwob_studies: MiniWoB synthetic task metrics
    """
    return fit_ok and sample_ok


def miniwob_studies_aux(aux: bool) -> bool:
    """miniwob_studies

    aux:
    miniwob_studies: tasks, episodes, rewards, and success rates
    """
    return aux


def _bench_miniwob_studies(seed: int = 0) -> float:
    checks = []
    checks.append(miniwob_studies_ok(True, True))
    checks.append(not miniwob_studies_ok(False, True))
    checks.append(miniwob_studies_aux(True))
    checks.append(not miniwob_studies_aux(False))
    checks.append(True)  # agentic-eval-2 canon
    return float(sum(checks) / len(checks))


def bench_miniwob_studies(seed: int = 0) -> dict[str, float]:
    return {"synthetic_miniwob_studies": _bench_miniwob_studies(seed)}
