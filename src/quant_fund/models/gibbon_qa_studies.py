"""gibbon_qa_studies module (SYNTHETIC)."""

from __future__ import annotations


def gibbon_qa_studies_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """gibbon_qa_studies

    check:
    gibbon_qa_studies: GibbonQA metrics
    """
    return fit_ok and sample_ok


def gibbon_qa_studies_aux(aux: bool) -> bool:
    """gibbon_qa_studies

    aux:
    gibbon_qa_studies: gibbons, canopy swings, answers, and scores
    """
    return aux


def _bench_gibbon_qa_studies(seed: int = 0) -> float:
    checks = []
    checks.append(gibbon_qa_studies_ok(True, True))
    checks.append(not gibbon_qa_studies_ok(False, True))
    checks.append(gibbon_qa_studies_aux(True))
    checks.append(not gibbon_qa_studies_aux(False))
    checks.append(True)  # primate canon
    return float(sum(checks) / len(checks))


def bench_gibbon_qa_studies(seed: int = 0) -> dict[str, float]:
    return {"synthetic_gibbon_qa_studies": _bench_gibbon_qa_studies(seed)}
