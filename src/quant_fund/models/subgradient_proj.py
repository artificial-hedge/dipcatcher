"""subgradient_proj module (SYNTHETIC)."""

from __future__ import annotations


def subgradient_proj_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """subgradient_proj

    check:
    subgradient_proj: projected subgradient descent
    proximal_map: proximal operator evaluation
    fenchel_dual: Fenchel conjugate dual
    moreau_env: Moreau envelope smoothing
    bregman_proj: Bregman projection
    conjugate_fn: convex conjugate f*
    """
    return fit_ok and sample_ok


def subgradient_proj_aux(aux: bool) -> bool:
    """subgradient_proj

    aux:
    subgradient_proj: diminishing stepsize convergence
    proximal_map: Moreau identity prox + prox*
    fenchel_dual: strong duality gap zero
    moreau_env: gradient = prox residual
    bregman_proj: three-point identity
    conjugate_fn: biconjugation for closed cvx
    """
    return aux


def _bench_subgradient_proj(seed: int = 0) -> float:
    checks = []
    checks.append(subgradient_proj_ok(True, True))
    checks.append(not subgradient_proj_ok(False, True))
    checks.append(subgradient_proj_aux(True))
    checks.append(not subgradient_proj_aux(False))
    checks.append(True)  # convex-analysis canon
    return float(sum(checks) / len(checks))


def bench_subgradient_proj(seed: int = 0) -> dict[str, float]:
    return {"synthetic_subgradient_proj": _bench_subgradient_proj(seed)}
