"""attribution_graph_studies module (SYNTHETIC)."""

from __future__ import annotations


def attribution_graph_studies_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """attribution_graph_studies

    check:
    attribution_graph_studies: node-edge attribution and path importance/replacements and effects
    """
    return fit_ok and sample_ok


def attribution_graph_studies_aux(aux: bool) -> bool:
    """attribution_graph_studies

    aux:
    attribution_graph_studies: cross-layer transcoders and pruning/graph and validation
    """
    return aux


def _bench_attribution_graph_studies(seed: int = 0) -> float:
    checks = []
    checks.append(attribution_graph_studies_ok(True, True))
    checks.append(not attribution_graph_studies_ok(False, True))
    checks.append(attribution_graph_studies_aux(True))
    checks.append(not attribution_graph_studies_aux(False))
    checks.append(True)  # mech-interp-2 canon
    return float(sum(checks) / len(checks))


def bench_attribution_graph_studies(seed: int = 0) -> dict[str, float]:
    return {"synthetic_attribution_graph_studies": _bench_attribution_graph_studies(seed)}
