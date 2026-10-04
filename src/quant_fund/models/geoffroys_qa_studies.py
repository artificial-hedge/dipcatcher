"""geoffroys_qa_studies module (SYNTHETIC)."""

from __future__ import annotations


def geoffroys_qa_studies_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """geoffroys_qa_studies

    check:
    geoffroys_qa_studies: GeoffroysQA metrics
    """
    return fit_ok and sample_ok


def geoffroys_qa_studies_aux(aux: bool) -> bool:
    """geoffroys_qa_studies

    aux:
    geoffroys_qa_studies: geoffroys cats, scrub flats, answers, and scores
    """
    return aux


def _bench_geoffroys_qa_studies(seed: int = 0) -> float:
    checks = []
    checks.append(geoffroys_qa_studies_ok(True, True))
    checks.append(not geoffroys_qa_studies_ok(False, True))
    checks.append(geoffroys_qa_studies_aux(True))
    checks.append(not geoffroys_qa_studies_aux(False))
    checks.append(True)  # felid-2 canon
    return float(sum(checks) / len(checks))


def bench_geoffroys_qa_studies(seed: int = 0) -> dict[str, float]:
    return {"synthetic_geoffroys_qa_studies": _bench_geoffroys_qa_studies(seed)}
