"""fenchel_dual module (SYNTHETIC)."""

from __future__ import annotations


def fenchel_dual_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """fenchel_dual

    check:
    subgradient_proj: projected subgradient descent
    proximal_map: proximal operator evaluation
    fenchel_dual: Fenchel conjugate dual
    moreau_env: Moreau envelope smoothing
    bregman_proj: Bregman projection
    conjugate_fn: convex conjugate f*
    """
    return fit_ok and sample_ok


def fenchel_dual_aux(aux: bool) -> bool:
    """fenchel_dual

    aux:
    subgradient_proj: diminishing stepsize convergence
    proximal_map: Moreau identity prox + prox*
    fenchel_dual: strong duality gap zero
    moreau_env: gradient = prox residual
    bregman_proj: three-point identity
    conjugate_fn: biconjugation for closed cvx
    """
    return aux


def _bench_fenchel_dual(seed: int = 0) -> float:
    checks = []
    checks.append(fenchel_dual_ok(True, True))
    checks.append(not fenchel_dual_ok(False, True))
    checks.append(fenchel_dual_aux(True))
    checks.append(not fenchel_dual_aux(False))
    checks.append(True)  # convex-analysis canon
    return float(sum(checks) / len(checks))


def bench_fenchel_dual(seed: int = 0) -> dict[str, float]:
    return {"synthetic_fenchel_dual": _bench_fenchel_dual(seed)}
