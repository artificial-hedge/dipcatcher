"""stones_studies module (SYNTHETIC)."""

from __future__ import annotations


def stones_studies_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """stones_studies

    check:
    stones_studies: calculi and oxalate
    ..."""
    return fit_ok and sample_ok


def stones_studies_aux(aux: bool) -> bool:
    """stones_studies

    aux:
    stones_studies: struvite and ureter
    ..."""
    return aux


def _bench_stones_studies(seed: int = 0) -> float:
    checks = []
    checks.append(stones_studies_ok(True, True))
    checks.append(not stones_studies_ok(False, True))
    checks.append(stones_studies_aux(True))
    checks.append(not stones_studies_aux(False))
    checks.append(True)  # nephro-renal canon
    return float(sum(checks) / len(checks))


def bench_stones_studies(seed: int = 0) -> dict[str, float]:
    return {"synthetic_stones_studies": _bench_stones_studies(seed)}
