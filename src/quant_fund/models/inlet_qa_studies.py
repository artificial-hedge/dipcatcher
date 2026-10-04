"""inlet_qa_studies module (SYNTHETIC)."""

from __future__ import annotations


def inlet_qa_studies_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """inlet_qa_studies

    check:
    inlet_qa_studies: InletQA metrics
    """
    return fit_ok and sample_ok


def inlet_qa_studies_aux(aux: bool) -> bool:
    """inlet_qa_studies

    aux:
    inlet_qa_studies: inlets, channels, answers, and scores
    """
    return aux


def _bench_inlet_qa_studies(seed: int = 0) -> float:
    checks = []
    checks.append(inlet_qa_studies_ok(True, True))
    checks.append(not inlet_qa_studies_ok(False, True))
    checks.append(inlet_qa_studies_aux(True))
    checks.append(not inlet_qa_studies_aux(False))
    checks.append(True)  # coastal canon
    return float(sum(checks) / len(checks))


def bench_inlet_qa_studies(seed: int = 0) -> dict[str, float]:
    return {"synthetic_inlet_qa_studies": _bench_inlet_qa_studies(seed)}
