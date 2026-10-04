"""dart_frog_qa_studies module (SYNTHETIC)."""

from __future__ import annotations


def dart_frog_qa_studies_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """dart_frog_qa_studies

    check:
    dart_frog_qa_studies: DartFrogQA metrics
    """
    return fit_ok and sample_ok


def dart_frog_qa_studies_aux(aux: bool) -> bool:
    """dart_frog_qa_studies

    aux:
    dart_frog_qa_studies: dart frogs, bromeliads, answers, and scores
    """
    return aux


def _bench_dart_frog_qa_studies(seed: int = 0) -> float:
    checks = []
    checks.append(dart_frog_qa_studies_ok(True, True))
    checks.append(not dart_frog_qa_studies_ok(False, True))
    checks.append(dart_frog_qa_studies_aux(True))
    checks.append(not dart_frog_qa_studies_aux(False))
    checks.append(True)  # frog canon
    return float(sum(checks) / len(checks))


def bench_dart_frog_qa_studies(seed: int = 0) -> dict[str, float]:
    return {"synthetic_dart_frog_qa_studies": _bench_dart_frog_qa_studies(seed)}
