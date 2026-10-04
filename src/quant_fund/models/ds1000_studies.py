"""ds1000_studies module (SYNTHETIC)."""

from __future__ import annotations


def ds1000_studies_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """ds1000_studies

    check:
    ds1000_studies: DS-1000 data-science snippet correctness metrics
    """
    return fit_ok and sample_ok


def ds1000_studies_aux(aux: bool) -> bool:
    """ds1000_studies

    aux:
    ds1000_studies: numpy/pandas tasks, tests, and scores
    """
    return aux


def _bench_ds1000_studies(seed: int = 0) -> float:
    checks = []
    checks.append(ds1000_studies_ok(True, True))
    checks.append(not ds1000_studies_ok(False, True))
    checks.append(ds1000_studies_aux(True))
    checks.append(not ds1000_studies_aux(False))
    checks.append(True)  # code-eval canon
    return float(sum(checks) / len(checks))


def bench_ds1000_studies(seed: int = 0) -> dict[str, float]:
    return {"synthetic_ds1000_studies": _bench_ds1000_studies(seed)}
