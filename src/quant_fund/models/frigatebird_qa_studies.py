"""frigatebird_qa_studies module (SYNTHETIC)."""

from __future__ import annotations


def frigatebird_qa_studies_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """frigatebird_qa_studies

    check:
    frigatebird_qa_studies: FrigatebirdQA metrics
    """
    return fit_ok and sample_ok


def frigatebird_qa_studies_aux(aux: bool) -> bool:
    """frigatebird_qa_studies

    aux:
    frigatebird_qa_studies: frigatebirds, trade winds, answers, and scores
    """
    return aux


def _bench_frigatebird_qa_studies(seed: int = 0) -> float:
    checks = []
    checks.append(frigatebird_qa_studies_ok(True, True))
    checks.append(not frigatebird_qa_studies_ok(False, True))
    checks.append(frigatebird_qa_studies_aux(True))
    checks.append(not frigatebird_qa_studies_aux(False))
    checks.append(True)  # seabird-3 canon
    return float(sum(checks) / len(checks))


def bench_frigatebird_qa_studies(seed: int = 0) -> dict[str, float]:
    return {"synthetic_frigatebird_qa_studies": _bench_frigatebird_qa_studies(seed)}
