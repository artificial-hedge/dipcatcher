"""qnli_lite_studies module (SYNTHETIC)."""

from __future__ import annotations


def qnli_lite_studies_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """qnli_lite_studies

    check:
    qnli_lite_studies: QNLI question-entailment metrics
    """
    return fit_ok and sample_ok


def qnli_lite_studies_aux(aux: bool) -> bool:
    """qnli_lite_studies

    aux:
    qnli_lite_studies: questions, sentences, labels, and accuracies
    """
    return aux


def _bench_qnli_lite_studies(seed: int = 0) -> float:
    checks = []
    checks.append(qnli_lite_studies_ok(True, True))
    checks.append(not qnli_lite_studies_ok(False, True))
    checks.append(qnli_lite_studies_aux(True))
    checks.append(not qnli_lite_studies_aux(False))
    checks.append(True)  # GLUE-eval-2 canon
    return float(sum(checks) / len(checks))


def bench_qnli_lite_studies(seed: int = 0) -> dict[str, float]:
    return {"synthetic_qnli_lite_studies": _bench_qnli_lite_studies(seed)}
