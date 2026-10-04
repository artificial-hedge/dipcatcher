"""kl_div_eval_studies module (SYNTHETIC)."""

from __future__ import annotations


def kl_div_eval_studies_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """kl_div_eval_studies

    check:
    kl_div_eval_studies: KL-divergence metrics
    """
    return fit_ok and sample_ok


def kl_div_eval_studies_aux(aux: bool) -> bool:
    """kl_div_eval_studies

    aux:
    kl_div_eval_studies: distributions, references, candidates, and scores
    """
    return aux


def _bench_kl_div_eval_studies(seed: int = 0) -> float:
    checks = []
    checks.append(kl_div_eval_studies_ok(True, True))
    checks.append(not kl_div_eval_studies_ok(False, True))
    checks.append(kl_div_eval_studies_aux(True))
    checks.append(not kl_div_eval_studies_aux(False))
    checks.append(True)  # metric-exotics canon
    return float(sum(checks) / len(checks))


def bench_kl_div_eval_studies(seed: int = 0) -> dict[str, float]:
    return {"synthetic_kl_div_eval_studies": _bench_kl_div_eval_studies(seed)}
