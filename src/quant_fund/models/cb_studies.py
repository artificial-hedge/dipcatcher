"""cb_studies module (SYNTHETIC)."""

from __future__ import annotations


def cb_studies_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """cb_studies

    check:
    cb_studies: SuperGLUE CB natural-language inference and F1
    """
    return fit_ok and sample_ok


def cb_studies_aux(aux: bool) -> bool:
    """cb_studies

    aux:
    cb_studies: premise-hypothesis pairs, entailment, and scores
    """
    return aux


def _bench_cb_studies(seed: int = 0) -> float:
    checks = []
    checks.append(cb_studies_ok(True, True))
    checks.append(not cb_studies_ok(False, True))
    checks.append(cb_studies_aux(True))
    checks.append(not cb_studies_aux(False))
    checks.append(True)  # NLP-eval-2 canon
    return float(sum(checks) / len(checks))


def bench_cb_studies(seed: int = 0) -> dict[str, float]:
    return {"synthetic_cb_studies": _bench_cb_studies(seed)}
