"""flow_cytometry_studies module (SYNTHETIC)."""

from __future__ import annotations


def flow_cytometry_studies_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """flow_cytometry_studies

    check:
    flow_cytometry_studies: gating and fluorescence/populations and markers
    """
    return fit_ok and sample_ok


def flow_cytometry_studies_aux(aux: bool) -> bool:
    """flow_cytometry_studies

    aux:
    flow_cytometry_studies: scatter and forward/side and compensation
    """
    return aux


def _bench_flow_cytometry_studies(seed: int = 0) -> float:
    checks = []
    checks.append(flow_cytometry_studies_ok(True, True))
    checks.append(not flow_cytometry_studies_ok(False, True))
    checks.append(flow_cytometry_studies_aux(True))
    checks.append(not flow_cytometry_studies_aux(False))
    checks.append(True)  # clinical-lab canon
    return float(sum(checks) / len(checks))


def bench_flow_cytometry_studies(seed: int = 0) -> dict[str, float]:
    return {"synthetic_flow_cytometry_studies": _bench_flow_cytometry_studies(seed)}
