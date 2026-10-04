"""prism_mt_studies module (SYNTHETIC)."""

from __future__ import annotations


def prism_mt_studies_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """prism_mt_studies

    check:
    prism_mt_studies: Prism MT-eval metrics
    """
    return fit_ok and sample_ok


def prism_mt_studies_aux(aux: bool) -> bool:
    """prism_mt_studies

    aux:
    prism_mt_studies: sources, hypotheses, references, and scores
    """
    return aux


def _bench_prism_mt_studies(seed: int = 0) -> float:
    checks = []
    checks.append(prism_mt_studies_ok(True, True))
    checks.append(not prism_mt_studies_ok(False, True))
    checks.append(prism_mt_studies_aux(True))
    checks.append(not prism_mt_studies_aux(False))
    checks.append(True)  # translation-metric canon
    return float(sum(checks) / len(checks))


def bench_prism_mt_studies(seed: int = 0) -> dict[str, float]:
    return {"synthetic_prism_mt_studies": _bench_prism_mt_studies(seed)}
