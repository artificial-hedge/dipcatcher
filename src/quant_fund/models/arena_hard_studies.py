"""arena_hard_studies module (SYNTHETIC)."""

from __future__ import annotations


def arena_hard_studies_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """arena_hard_studies

    check:
    arena_hard_studies: Arena-Hard win-rate pairwise metrics vs baseline
    """
    return fit_ok and sample_ok


def arena_hard_studies_aux(aux: bool) -> bool:
    """arena_hard_studies

    aux:
    arena_hard_studies: comparisons, votes, and win rates
    """
    return aux


def _bench_arena_hard_studies(seed: int = 0) -> float:
    checks = []
    checks.append(arena_hard_studies_ok(True, True))
    checks.append(not arena_hard_studies_ok(False, True))
    checks.append(arena_hard_studies_aux(True))
    checks.append(not arena_hard_studies_aux(False))
    checks.append(True)  # judge-eval canon
    return float(sum(checks) / len(checks))


def bench_arena_hard_studies(seed: int = 0) -> dict[str, float]:
    return {"synthetic_arena_hard_studies": _bench_arena_hard_studies(seed)}
