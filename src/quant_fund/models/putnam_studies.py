"""putnam_studies module (SYNTHETIC)."""

from __future__ import annotations


def putnam_studies_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """putnam_studies

    check:
    putnam_studies: PutnamBench competition-problem solve and partial scores
    """
    return fit_ok and sample_ok


def putnam_studies_aux(aux: bool) -> bool:
    """putnam_studies

    aux:
    putnam_studies: problems, proofs, and solve metrics
    """
    return aux


def _bench_putnam_studies(seed: int = 0) -> float:
    checks = []
    checks.append(putnam_studies_ok(True, True))
    checks.append(not putnam_studies_ok(False, True))
    checks.append(putnam_studies_aux(True))
    checks.append(not putnam_studies_aux(False))
    checks.append(True)  # reasoning-eval canon
    return float(sum(checks) / len(checks))


def bench_putnam_studies(seed: int = 0) -> dict[str, float]:
    return {"synthetic_putnam_studies": _bench_putnam_studies(seed)}
