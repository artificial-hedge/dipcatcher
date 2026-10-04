"""function_vector_studies module (SYNTHETIC)."""

from __future__ import annotations


def function_vector_studies_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """function_vector_studies

    check:
    function_vector_studies: task representations in activation space/averaging and steering
    """
    return fit_ok and sample_ok


def function_vector_studies_aux(aux: bool) -> bool:
    """function_vector_studies

    aux:
    function_vector_studies: ICL vectors and transport/heads and directions
    """
    return aux


def _bench_function_vector_studies(seed: int = 0) -> float:
    checks = []
    checks.append(function_vector_studies_ok(True, True))
    checks.append(not function_vector_studies_ok(False, True))
    checks.append(function_vector_studies_aux(True))
    checks.append(not function_vector_studies_aux(False))
    checks.append(True)  # interpretability-3 canon
    return float(sum(checks) / len(checks))


def bench_function_vector_studies(seed: int = 0) -> dict[str, float]:
    return {"synthetic_function_vector_studies": _bench_function_vector_studies(seed)}
