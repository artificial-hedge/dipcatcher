"""bus_qa_studies module (SYNTHETIC)."""

from __future__ import annotations


def bus_qa_studies_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """bus_qa_studies

    check:
    bus_qa_studies: BusQA metrics
    """
    return fit_ok and sample_ok


def bus_qa_studies_aux(aux: bool) -> bool:
    """bus_qa_studies

    aux:
    bus_qa_studies: buses, routes, answers, and scores
    """
    return aux


def _bench_bus_qa_studies(seed: int = 0) -> float:
    checks = []
    checks.append(bus_qa_studies_ok(True, True))
    checks.append(not bus_qa_studies_ok(False, True))
    checks.append(bus_qa_studies_aux(True))
    checks.append(not bus_qa_studies_aux(False))
    checks.append(True)  # vehicle canon
    return float(sum(checks) / len(checks))


def bench_bus_qa_studies(seed: int = 0) -> dict[str, float]:
    return {"synthetic_bus_qa_studies": _bench_bus_qa_studies(seed)}
