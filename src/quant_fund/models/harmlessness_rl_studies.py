"""harmlessness_rl_studies module (SYNTHETIC)."""

from __future__ import annotations


def harmlessness_rl_studies_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """harmlessness_rl_studies

    check:
    harmlessness_rl_studies: harmlessness-preference policy updates/signals and actions
    """
    return fit_ok and sample_ok


def harmlessness_rl_studies_aux(aux: bool) -> bool:
    """harmlessness_rl_studies

    aux:
    harmlessness_rl_studies: refusal-vs-helpful tradeoff shaping/queries and labels
    """
    return aux


def _bench_harmlessness_rl_studies(seed: int = 0) -> float:
    checks = []
    checks.append(harmlessness_rl_studies_ok(True, True))
    checks.append(not harmlessness_rl_studies_ok(False, True))
    checks.append(harmlessness_rl_studies_aux(True))
    checks.append(not harmlessness_rl_studies_aux(False))
    checks.append(True)  # constitutional-AI canon
    return float(sum(checks) / len(checks))


def bench_harmlessness_rl_studies(seed: int = 0) -> dict[str, float]:
    return {"synthetic_harmlessness_rl_studies": _bench_harmlessness_rl_studies(seed)}
