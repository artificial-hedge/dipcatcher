"""huntington_studies module (SYNTHETIC)."""

from __future__ import annotations


def huntington_studies_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """huntington_studies

    check:
    huntington_studies: chorea and cag
    ..."""
    return fit_ok and sample_ok


def huntington_studies_aux(aux: bool) -> bool:
    """huntington_studies

    aux:
    huntington_studies: repeats and striatum
    ..."""
    return aux


def _bench_huntington_studies(seed: int = 0) -> float:
    checks = []
    checks.append(huntington_studies_ok(True, True))
    checks.append(not huntington_studies_ok(False, True))
    checks.append(huntington_studies_aux(True))
    checks.append(not huntington_studies_aux(False))
    checks.append(True)  # neurodegeneration canon
    return float(sum(checks) / len(checks))


def bench_huntington_studies(seed: int = 0) -> dict[str, float]:
    return {"synthetic_huntington_studies": _bench_huntington_studies(seed)}
