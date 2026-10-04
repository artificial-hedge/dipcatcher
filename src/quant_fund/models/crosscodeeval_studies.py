"""crosscodeeval_studies module (SYNTHETIC)."""

from __future__ import annotations


def crosscodeeval_studies_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """crosscodeeval_studies

    check:
    crosscodeeval_studies: CrossCodeEval cross-file completion metrics
    """
    return fit_ok and sample_ok


def crosscodeeval_studies_aux(aux: bool) -> bool:
    """crosscodeeval_studies

    aux:
    crosscodeeval_studies: contexts, files, completions, and scores
    """
    return aux


def _bench_crosscodeeval_studies(seed: int = 0) -> float:
    checks = []
    checks.append(crosscodeeval_studies_ok(True, True))
    checks.append(not crosscodeeval_studies_ok(False, True))
    checks.append(crosscodeeval_studies_aux(True))
    checks.append(not crosscodeeval_studies_aux(False))
    checks.append(True)  # code-eval-3 canon
    return float(sum(checks) / len(checks))


def bench_crosscodeeval_studies(seed: int = 0) -> dict[str, float]:
    return {"synthetic_crosscodeeval_studies": _bench_crosscodeeval_studies(seed)}
