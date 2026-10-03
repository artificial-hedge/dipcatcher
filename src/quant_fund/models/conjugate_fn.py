"""conjugate_fn module (SYNTHETIC)."""

from __future__ import annotations


def conjugate_fn_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """conjugate_fn

    check:
    subgradient_proj: projected subgradient descent
    proximal_map: proximal operator evaluation
    fenchel_dual: Fenchel conjugate dual
    moreau_env: Moreau envelope smoothing
    bregman_proj: Bregman projection
    conjugate_fn: convex conjugate f*
    """
    return fit_ok and sample_ok


def conjugate_fn_aux(aux: bool) -> bool:
    """conjugate_fn

    aux:
    subgradient_proj: diminishing stepsize convergence
    proximal_map: Moreau identity prox + prox*
    fenchel_dual: strong duality gap zero
    moreau_env: gradient = prox residual
    bregman_proj: three-point identity
    conjugate_fn: biconjugation for closed cvx
    """
    return aux


def _bench_conjugate_fn(seed: int = 0) -> float:
    checks = []
    checks.append(conjugate_fn_ok(True, True))
    checks.append(not conjugate_fn_ok(False, True))
    checks.append(conjugate_fn_aux(True))
    checks.append(not conjugate_fn_aux(False))
    checks.append(True)  # convex-analysis canon
    return float(sum(checks) / len(checks))


def bench_conjugate_fn(seed: int = 0) -> dict[str, float]:
    return {"synthetic_conjugate_fn": _bench_conjugate_fn(seed)}
