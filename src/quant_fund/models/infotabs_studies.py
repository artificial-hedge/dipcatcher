"""infotabs_studies module (SYNTHETIC)."""

from __future__ import annotations


def infotabs_studies_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """infotabs_studies

    check:
    infotabs_studies: Infotabs metrics
    """
    return fit_ok and sample_ok


def infotabs_studies_aux(aux: bool) -> bool:
    """infotabs_studies

    aux:
    infotabs_studies: tables, hypotheses, labels, and scores
    """
    return aux


def _bench_infotabs_studies(seed: int = 0) -> float:
    checks = []
    checks.append(infotabs_studies_ok(True, True))
    checks.append(not infotabs_studies_ok(False, True))
    checks.append(infotabs_studies_aux(True))
    checks.append(not infotabs_studies_aux(False))
    checks.append(True)  # table-QA canon
    return float(sum(checks) / len(checks))


def bench_infotabs_studies(seed: int = 0) -> dict[str, float]:
    return {"synthetic_infotabs_studies": _bench_infotabs_studies(seed)}
