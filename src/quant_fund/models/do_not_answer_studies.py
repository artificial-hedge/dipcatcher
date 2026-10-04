"""do_not_answer_studies module (SYNTHETIC)."""

from __future__ import annotations


def do_not_answer_studies_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """do_not_answer_studies

    check:
    do_not_answer_studies: Do-Not-Answer responsible-refusal metrics
    """
    return fit_ok and sample_ok


def do_not_answer_studies_aux(aux: bool) -> bool:
    """do_not_answer_studies

    aux:
    do_not_answer_studies: questions, refusals, categories, and rates
    """
    return aux


def _bench_do_not_answer_studies(seed: int = 0) -> float:
    checks = []
    checks.append(do_not_answer_studies_ok(True, True))
    checks.append(not do_not_answer_studies_ok(False, True))
    checks.append(do_not_answer_studies_aux(True))
    checks.append(not do_not_answer_studies_aux(False))
    checks.append(True)  # safety-alignment-2 canon
    return float(sum(checks) / len(checks))


def bench_do_not_answer_studies(seed: int = 0) -> dict[str, float]:
    return {"synthetic_do_not_answer_studies": _bench_do_not_answer_studies(seed)}
