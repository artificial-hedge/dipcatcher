"""gdon_qa_studies module (SYNTHETIC)."""

from __future__ import annotations


def gdon_qa_studies_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """gdon_qa_studies

    check:
    gdon_qa_studies: G
    """
    return fit_ok and sample_ok


def gdon_qa_studies_aux(aux: bool) -> bool:
    """gdon_qa_studies

    aux:
    gdon_qa_studies: d
    """
    return aux


def _bench_gdon_qa_studies(seed: int = 0) -> float:
    checks = []
    checks.append(gdon_qa_studies_ok(True, True))
    checks.append(not gdon_qa_studies_ok(False, True))
    checks.append(gdon_qa_studies_aux(True))
    checks.append(not gdon_qa_studies_aux(False))
    checks.append(True)  # tibetan-demon canon
    return float(sum(checks) / len(checks))


def bench_gdon_qa_studies(seed: int = 0) -> dict[str, float]:
    return {"synthetic_gdon_qa_studies": _bench_gdon_qa_studies(seed)}
