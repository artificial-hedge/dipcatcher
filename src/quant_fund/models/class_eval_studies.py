"""class_eval_studies module (SYNTHETIC)."""

from __future__ import annotations


def class_eval_studies_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """class_eval_studies

    check:
    class_eval_studies: ClassEval class-level generation metrics
    """
    return fit_ok and sample_ok


def class_eval_studies_aux(aux: bool) -> bool:
    """class_eval_studies

    aux:
    class_eval_studies: classes, methods, and pass rates
    """
    return aux


def _bench_class_eval_studies(seed: int = 0) -> float:
    checks = []
    checks.append(class_eval_studies_ok(True, True))
    checks.append(not class_eval_studies_ok(False, True))
    checks.append(class_eval_studies_aux(True))
    checks.append(not class_eval_studies_aux(False))
    checks.append(True)  # code-eval-2 canon
    return float(sum(checks) / len(checks))


def bench_class_eval_studies(seed: int = 0) -> dict[str, float]:
    return {"synthetic_class_eval_studies": _bench_class_eval_studies(seed)}
