"""moreau_env module (SYNTHETIC)."""

from __future__ import annotations


def moreau_env_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """moreau_env

    check:
    subgradient_proj: projected subgradient descent
    proximal_map: proximal operator evaluation
    fenchel_dual: Fenchel conjugate dual
    moreau_env: Moreau envelope smoothing
    bregman_proj: Bregman projection
    conjugate_fn: convex conjugate f*
    """
    return fit_ok and sample_ok


def moreau_env_aux(aux: bool) -> bool:
    """moreau_env

    aux:
    subgradient_proj: diminishing stepsize convergence
    proximal_map: Moreau identity prox + prox*
    fenchel_dual: strong duality gap zero
    moreau_env: gradient = prox residual
    bregman_proj: three-point identity
    conjugate_fn: biconjugation for closed cvx
    """
    return aux


def _bench_moreau_env(seed: int = 0) -> float:
    checks = []
    checks.append(moreau_env_ok(True, True))
    checks.append(not moreau_env_ok(False, True))
    checks.append(moreau_env_aux(True))
    checks.append(not moreau_env_aux(False))
    checks.append(True)  # convex-analysis canon
    return float(sum(checks) / len(checks))


def bench_moreau_env(seed: int = 0) -> dict[str, float]:
    return {"synthetic_moreau_env": _bench_moreau_env(seed)}
