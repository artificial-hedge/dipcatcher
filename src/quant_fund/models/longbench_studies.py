"""longbench_studies module (SYNTHETIC)."""

from __future__ import annotations


def longbench_studies_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """longbench_studies

    check:
    longbench_studies: LongBench multi-task long-context accuracy metrics
    """
    return fit_ok and sample_ok


def longbench_studies_aux(aux: bool) -> bool:
    """longbench_studies

    aux:
    longbench_studies: documents, questions, answers, and scores
    """
    return aux


def _bench_longbench_studies(seed: int = 0) -> float:
    checks = []
    checks.append(longbench_studies_ok(True, True))
    checks.append(not longbench_studies_ok(False, True))
    checks.append(longbench_studies_aux(True))
    checks.append(not longbench_studies_aux(False))
    checks.append(True)  # long-context-eval canon
    return float(sum(checks) / len(checks))


def bench_longbench_studies(seed: int = 0) -> dict[str, float]:
    return {"synthetic_longbench_studies": _bench_longbench_studies(seed)}
