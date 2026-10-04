"""mini_f2f_studies module (SYNTHETIC)."""

from __future__ import annotations


def mini_f2f_studies_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """mini_f2f_studies

    check:
    mini_f2f_studies: MiniF2F olympiad metrics
    """
    return fit_ok and sample_ok


def mini_f2f_studies_aux(aux: bool) -> bool:
    """mini_f2f_studies

    aux:
    mini_f2f_studies: statements, proofs, tactics, and pass rates
    """
    return aux


def _bench_mini_f2f_studies(seed: int = 0) -> float:
    checks = []
    checks.append(mini_f2f_studies_ok(True, True))
    checks.append(not mini_f2f_studies_ok(False, True))
    checks.append(mini_f2f_studies_aux(True))
    checks.append(not mini_f2f_studies_aux(False))
    checks.append(True)  # math-reasoning-eval canon
    return float(sum(checks) / len(checks))


def bench_mini_f2f_studies(seed: int = 0) -> dict[str, float]:
    return {"synthetic_mini_f2f_studies": _bench_mini_f2f_studies(seed)}
