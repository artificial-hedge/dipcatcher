"""sppo_studies module (SYNTHETIC)."""

from __future__ import annotations


def sppo_studies_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """sppo_studies

    check:
    sppo_studies: self-play preference and game-theoretic losses/iterations and approximations
    """
    return fit_ok and sample_ok


def sppo_studies_aux(aux: bool) -> bool:
    """sppo_studies

    aux:
    sppo_studies: symmetric updates and convergence/estimation and equilibrium
    """
    return aux


def _bench_sppo_studies(seed: int = 0) -> float:
    checks = []
    checks.append(sppo_studies_ok(True, True))
    checks.append(not sppo_studies_ok(False, True))
    checks.append(sppo_studies_aux(True))
    checks.append(not sppo_studies_aux(False))
    checks.append(True)  # post-training-2 canon
    return float(sum(checks) / len(checks))


def bench_sppo_studies(seed: int = 0) -> dict[str, float]:
    return {"synthetic_sppo_studies": _bench_sppo_studies(seed)}
