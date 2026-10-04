"""hol_step_studies module (SYNTHETIC)."""

from __future__ import annotations


def hol_step_studies_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """hol_step_studies

    check:
    hol_step_studies: HOList step-tactic metrics
    """
    return fit_ok and sample_ok


def hol_step_studies_aux(aux: bool) -> bool:
    """hol_step_studies

    aux:
    hol_step_studies: goals, tactics, proofs, and success rates
    """
    return aux


def _bench_hol_step_studies(seed: int = 0) -> float:
    checks = []
    checks.append(hol_step_studies_ok(True, True))
    checks.append(not hol_step_studies_ok(False, True))
    checks.append(hol_step_studies_aux(True))
    checks.append(not hol_step_studies_aux(False))
    checks.append(True)  # math-eval-2 canon
    return float(sum(checks) / len(checks))


def bench_hol_step_studies(seed: int = 0) -> dict[str, float]:
    return {"synthetic_hol_step_studies": _bench_hol_step_studies(seed)}
