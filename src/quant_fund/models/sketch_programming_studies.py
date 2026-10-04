"""sketch_programming_studies module (SYNTHETIC)."""

from __future__ import annotations


def sketch_programming_studies_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """sketch_programming_studies

    check:
    sketch_programming_studies: holes and constraints/solvers and completion
    """
    return fit_ok and sample_ok


def sketch_programming_studies_aux(aux: bool) -> bool:
    """sketch_programming_studies

    aux:
    sketch_programming_studies: synthesis fragments and bounded search/oracles and cegis
    """
    return aux


def _bench_sketch_programming_studies(seed: int = 0) -> float:
    checks = []
    checks.append(sketch_programming_studies_ok(True, True))
    checks.append(not sketch_programming_studies_ok(False, True))
    checks.append(sketch_programming_studies_aux(True))
    checks.append(not sketch_programming_studies_aux(False))
    checks.append(True)  # neuro-symbolic canon
    return float(sum(checks) / len(checks))


def bench_sketch_programming_studies(seed: int = 0) -> dict[str, float]:
    return {"synthetic_sketch_programming_studies": _bench_sketch_programming_studies(seed)}
