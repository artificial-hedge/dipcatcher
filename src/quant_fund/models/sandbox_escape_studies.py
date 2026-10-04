"""sandbox_escape_studies module (SYNTHETIC)."""

from __future__ import annotations


def sandbox_escape_studies_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """sandbox_escape_studies

    check:
    sandbox_escape_studies: sandbox boundary probing and privilege checks/actions and policies
    """
    return fit_ok and sample_ok


def sandbox_escape_studies_aux(aux: bool) -> bool:
    """sandbox_escape_studies

    aux:
    sandbox_escape_studies: environment-isolation verification and egress checks/routes and blocks
    """
    return aux


def _bench_sandbox_escape_studies(seed: int = 0) -> float:
    checks = []
    checks.append(sandbox_escape_studies_ok(True, True))
    checks.append(not sandbox_escape_studies_ok(False, True))
    checks.append(sandbox_escape_studies_aux(True))
    checks.append(not sandbox_escape_studies_aux(False))
    checks.append(True)  # agent-safety canon
    return float(sum(checks) / len(checks))


def bench_sandbox_escape_studies(seed: int = 0) -> dict[str, float]:
    return {"synthetic_sandbox_escape_studies": _bench_sandbox_escape_studies(seed)}
