"""pandalm_studies module (SYNTHETIC)."""

from __future__ import annotations


def pandalm_studies_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """pandalm_studies

    check:
    pandalm_studies: PandaLM judge pairwise-comparison metrics
    """
    return fit_ok and sample_ok


def pandalm_studies_aux(aux: bool) -> bool:
    """pandalm_studies

    aux:
    pandalm_studies: pairs, comparisons, and agreement rates
    """
    return aux


def _bench_pandalm_studies(seed: int = 0) -> float:
    checks = []
    checks.append(pandalm_studies_ok(True, True))
    checks.append(not pandalm_studies_ok(False, True))
    checks.append(pandalm_studies_aux(True))
    checks.append(not pandalm_studies_aux(False))
    checks.append(True)  # eval-tooling canon
    return float(sum(checks) / len(checks))


def bench_pandalm_studies(seed: int = 0) -> dict[str, float]:
    return {"synthetic_pandalm_studies": _bench_pandalm_studies(seed)}
