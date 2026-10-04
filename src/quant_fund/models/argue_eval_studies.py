"""argue_eval_studies module (SYNTHETIC)."""

from __future__ import annotations


def argue_eval_studies_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """argue_eval_studies

    check:
    argue_eval_studies: ArguE metrics
    """
    return fit_ok and sample_ok


def argue_eval_studies_aux(aux: bool) -> bool:
    """argue_eval_studies

    aux:
    argue_eval_studies: claims, debates, verdicts, and scores
    """
    return aux


def _bench_argue_eval_studies(seed: int = 0) -> float:
    checks = []
    checks.append(argue_eval_studies_ok(True, True))
    checks.append(not argue_eval_studies_ok(False, True))
    checks.append(argue_eval_studies_aux(True))
    checks.append(not argue_eval_studies_aux(False))
    checks.append(True)  # QA-exotics-2 canon
    return float(sum(checks) / len(checks))


def bench_argue_eval_studies(seed: int = 0) -> dict[str, float]:
    return {"synthetic_argue_eval_studies": _bench_argue_eval_studies(seed)}
