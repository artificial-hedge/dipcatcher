"""capuchin_qa_studies module (SYNTHETIC)."""

from __future__ import annotations


def capuchin_qa_studies_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """capuchin_qa_studies

    check:
    capuchin_qa_studies: CapuchinQA metrics
    """
    return fit_ok and sample_ok


def capuchin_qa_studies_aux(aux: bool) -> bool:
    """capuchin_qa_studies

    aux:
    capuchin_qa_studies: capuchins, dry forests, answers, and scores
    """
    return aux


def _bench_capuchin_qa_studies(seed: int = 0) -> float:
    checks = []
    checks.append(capuchin_qa_studies_ok(True, True))
    checks.append(not capuchin_qa_studies_ok(False, True))
    checks.append(capuchin_qa_studies_aux(True))
    checks.append(not capuchin_qa_studies_aux(False))
    checks.append(True)  # new-world-monkey canon
    return float(sum(checks) / len(checks))


def bench_capuchin_qa_studies(seed: int = 0) -> dict[str, float]:
    return {"synthetic_capuchin_qa_studies": _bench_capuchin_qa_studies(seed)}
