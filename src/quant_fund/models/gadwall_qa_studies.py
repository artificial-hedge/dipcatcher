"""gadwall_qa_studies module (SYNTHETIC)."""

from __future__ import annotations


def gadwall_qa_studies_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """gadwall_qa_studies

    check:
    gadwall_qa_studies: GadwallQA metrics
    """
    return fit_ok and sample_ok


def gadwall_qa_studies_aux(aux: bool) -> bool:
    """gadwall_qa_studies

    aux:
    gadwall_qa_studies: gadwalls, ponds, answers, and scores
    """
    return aux


def _bench_gadwall_qa_studies(seed: int = 0) -> float:
    checks = []
    checks.append(gadwall_qa_studies_ok(True, True))
    checks.append(not gadwall_qa_studies_ok(False, True))
    checks.append(gadwall_qa_studies_aux(True))
    checks.append(not gadwall_qa_studies_aux(False))
    checks.append(True)  # waterfowl canon
    return float(sum(checks) / len(checks))


def bench_gadwall_qa_studies(seed: int = 0) -> dict[str, float]:
    return {"synthetic_gadwall_qa_studies": _bench_gadwall_qa_studies(seed)}
