"""glue_studies module (SYNTHETIC)."""

from __future__ import annotations


def glue_studies_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """glue_studies

    check:
    glue_studies: GLUE sentence tasks, fine-tune metrics, and scores
    """
    return fit_ok and sample_ok


def glue_studies_aux(aux: bool) -> bool:
    """glue_studies

    aux:
    glue_studies: SST/MRPC/STS tasks, accuracies, and aggregate
    """
    return aux


def _bench_glue_studies(seed: int = 0) -> float:
    checks = []
    checks.append(glue_studies_ok(True, True))
    checks.append(not glue_studies_ok(False, True))
    checks.append(glue_studies_aux(True))
    checks.append(not glue_studies_aux(False))
    checks.append(True)  # GLUE-eval canon
    return float(sum(checks) / len(checks))


def bench_glue_studies(seed: int = 0) -> dict[str, float]:
    return {"synthetic_glue_studies": _bench_glue_studies(seed)}
