"""muin_lite_studies module (SYNTHETIC)."""

from __future__ import annotations


def muin_lite_studies_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """muin_lite_studies

    check:
    muin_lite_studies: MuIn sequential-reasoning metrics
    """
    return fit_ok and sample_ok


def muin_lite_studies_aux(aux: bool) -> bool:
    """muin_lite_studies

    aux:
    muin_lite_studies: episodes, states, actions, and scores
    """
    return aux


def _bench_muin_lite_studies(seed: int = 0) -> float:
    checks = []
    checks.append(muin_lite_studies_ok(True, True))
    checks.append(not muin_lite_studies_ok(False, True))
    checks.append(muin_lite_studies_aux(True))
    checks.append(not muin_lite_studies_aux(False))
    checks.append(True)  # commonsense-reasoning canon
    return float(sum(checks) / len(checks))


def bench_muin_lite_studies(seed: int = 0) -> dict[str, float]:
    return {"synthetic_muin_lite_studies": _bench_muin_lite_studies(seed)}
