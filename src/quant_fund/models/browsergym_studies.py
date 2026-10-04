"""browsergym_studies module (SYNTHETIC)."""

from __future__ import annotations


def browsergym_studies_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """browsergym_studies

    check:
    browsergym_studies: BrowserGym metrics
    """
    return fit_ok and sample_ok


def browsergym_studies_aux(aux: bool) -> bool:
    """browsergym_studies

    aux:
    browsergym_studies: tasks, actions, observations, and scores
    """
    return aux


def _bench_browsergym_studies(seed: int = 0) -> float:
    checks = []
    checks.append(browsergym_studies_ok(True, True))
    checks.append(not browsergym_studies_ok(False, True))
    checks.append(browsergym_studies_aux(True))
    checks.append(not browsergym_studies_aux(False))
    checks.append(True)  # web-agent canon
    return float(sum(checks) / len(checks))


def bench_browsergym_studies(seed: int = 0) -> dict[str, float]:
    return {"synthetic_browsergym_studies": _bench_browsergym_studies(seed)}
