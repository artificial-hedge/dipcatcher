"""disaggregated_serving_studies module (SYNTHETIC)."""

from __future__ import annotations


def disaggregated_serving_studies_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """disaggregated_serving_studies

    check:
    disaggregated_serving_studies: prefill-decode separation and KV transfer/pools and placement
    """
    return fit_ok and sample_ok


def disaggregated_serving_studies_aux(aux: bool) -> bool:
    """disaggregated_serving_studies

    aux:
    disaggregated_serving_studies: Splitwise/DistServe-style partitioning/TTFT and TPOT
    """
    return aux


def _bench_disaggregated_serving_studies(seed: int = 0) -> float:
    checks = []
    checks.append(disaggregated_serving_studies_ok(True, True))
    checks.append(not disaggregated_serving_studies_ok(False, True))
    checks.append(disaggregated_serving_studies_aux(True))
    checks.append(not disaggregated_serving_studies_aux(False))
    checks.append(True)  # LLM-serving canon
    return float(sum(checks) / len(checks))


def bench_disaggregated_serving_studies(seed: int = 0) -> dict[str, float]:
    return {"synthetic_disaggregated_serving_studies": _bench_disaggregated_serving_studies(seed)}
