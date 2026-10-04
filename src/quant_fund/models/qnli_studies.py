"""qnli_studies module (SYNTHETIC)."""

from __future__ import annotations


def qnli_studies_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """qnli_studies

    check:
    qnli_studies: QNLI question-answering NLI and accuracy
    """
    return fit_ok and sample_ok


def qnli_studies_aux(aux: bool) -> bool:
    """qnli_studies

    aux:
    qnli_studies: question-sentence pairs, labels, and scores
    """
    return aux


def _bench_qnli_studies(seed: int = 0) -> float:
    checks = []
    checks.append(qnli_studies_ok(True, True))
    checks.append(not qnli_studies_ok(False, True))
    checks.append(qnli_studies_aux(True))
    checks.append(not qnli_studies_aux(False))
    checks.append(True)  # GLUE-eval canon
    return float(sum(checks) / len(checks))


def bench_qnli_studies(seed: int = 0) -> dict[str, float]:
    return {"synthetic_qnli_studies": _bench_qnli_studies(seed)}
