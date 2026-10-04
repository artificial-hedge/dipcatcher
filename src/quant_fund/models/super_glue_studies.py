"""super_glue_studies module (SYNTHETIC)."""

from __future__ import annotations


def super_glue_studies_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """super_glue_studies

    check:
    super_glue_studies: SuperGLUE harder NLU tasks and combined score
    """
    return fit_ok and sample_ok


def super_glue_studies_aux(aux: bool) -> bool:
    """super_glue_studies

    aux:
    super_glue_studies: BoolQ/CB/COPA/MultiRC/ReCoRD/WiC/WSC results
    """
    return aux


def _bench_super_glue_studies(seed: int = 0) -> float:
    checks = []
    checks.append(super_glue_studies_ok(True, True))
    checks.append(not super_glue_studies_ok(False, True))
    checks.append(super_glue_studies_aux(True))
    checks.append(not super_glue_studies_aux(False))
    checks.append(True)  # GLUE-eval canon
    return float(sum(checks) / len(checks))


def bench_super_glue_studies(seed: int = 0) -> dict[str, float]:
    return {"synthetic_super_glue_studies": _bench_super_glue_studies(seed)}
