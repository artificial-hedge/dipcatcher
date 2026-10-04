"""win_ratio_studies module (SYNTHETIC)."""

from __future__ import annotations


def win_ratio_studies_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """win_ratio_studies

    check:
    win_ratio_studies: hierarchical endpoints/pairwise wins and ties
    """
    return fit_ok and sample_ok


def win_ratio_studies_aux(aux: bool) -> bool:
    """win_ratio_studies

    aux:
    win_ratio_studies: priority and censoring/handling and analysis
    """
    return aux


def _bench_win_ratio_studies(seed: int = 0) -> float:
    checks = []
    checks.append(win_ratio_studies_ok(True, True))
    checks.append(not win_ratio_studies_ok(False, True))
    checks.append(win_ratio_studies_aux(True))
    checks.append(not win_ratio_studies_aux(False))
    checks.append(True)  # target-trial/RWE canon
    return float(sum(checks) / len(checks))


def bench_win_ratio_studies(seed: int = 0) -> dict[str, float]:
    return {"synthetic_win_ratio_studies": _bench_win_ratio_studies(seed)}
