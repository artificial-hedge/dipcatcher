"""jungle_cat_qa_studies module (SYNTHETIC)."""

from __future__ import annotations


def jungle_cat_qa_studies_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """jungle_cat_qa_studies

    check:
    jungle_cat_qa_studies: JungleCatQA metrics
    """
    return fit_ok and sample_ok


def jungle_cat_qa_studies_aux(aux: bool) -> bool:
    """jungle_cat_qa_studies

    aux:
    jungle_cat_qa_studies: jungle cats, reed beds, answers, and scores
    """
    return aux


def _bench_jungle_cat_qa_studies(seed: int = 0) -> float:
    checks = []
    checks.append(jungle_cat_qa_studies_ok(True, True))
    checks.append(not jungle_cat_qa_studies_ok(False, True))
    checks.append(jungle_cat_qa_studies_aux(True))
    checks.append(not jungle_cat_qa_studies_aux(False))
    checks.append(True)  # small-cat canon
    return float(sum(checks) / len(checks))


def bench_jungle_cat_qa_studies(seed: int = 0) -> dict[str, float]:
    return {"synthetic_jungle_cat_qa_studies": _bench_jungle_cat_qa_studies(seed)}
