"""web_arena_studies module (SYNTHETIC)."""

from __future__ import annotations


def web_arena_studies_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """web_arena_studies

    check:
    web_arena_studies: WebArena site tasks/urls and success checks
    """
    return fit_ok and sample_ok


def web_arena_studies_aux(aux: bool) -> bool:
    """web_arena_studies

    aux:
    web_arena_studies: browser actions/obs and reward funcs
    """
    return aux


def _bench_web_arena_studies(seed: int = 0) -> float:
    checks = []
    checks.append(web_arena_studies_ok(True, True))
    checks.append(not web_arena_studies_ok(False, True))
    checks.append(web_arena_studies_aux(True))
    checks.append(not web_arena_studies_aux(False))
    checks.append(True)  # agentic-eval canon
    return float(sum(checks) / len(checks))


def bench_web_arena_studies(seed: int = 0) -> dict[str, float]:
    return {"synthetic_web_arena_studies": _bench_web_arena_studies(seed)}
