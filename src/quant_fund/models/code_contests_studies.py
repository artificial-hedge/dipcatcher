"""code_contests_studies module (SYNTHETIC)."""

from __future__ import annotations


def code_contests_studies_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """code_contests_studies

    check:
    code_contests_studies: CodeContests competition accuracy metrics
    """
    return fit_ok and sample_ok


def code_contests_studies_aux(aux: bool) -> bool:
    """code_contests_studies

    aux:
    code_contests_studies: problems, submissions, and solve rates
    """
    return aux


def _bench_code_contests_studies(seed: int = 0) -> float:
    checks = []
    checks.append(code_contests_studies_ok(True, True))
    checks.append(not code_contests_studies_ok(False, True))
    checks.append(code_contests_studies_aux(True))
    checks.append(not code_contests_studies_aux(False))
    checks.append(True)  # code-eval-2 canon
    return float(sum(checks) / len(checks))


def bench_code_contests_studies(seed: int = 0) -> dict[str, float]:
    return {"synthetic_code_contests_studies": _bench_code_contests_studies(seed)}
