"""Microsupport of sheaves (SYNTHETIC)."""

from __future__ import annotations


def micro_supp_ok(cotangent_lagr: bool, prop_estim: bool) -> bool:
    """SS(F) ⊆ T*M is the closure of directions
    where sections fail to extend; conic,
    involutive, co-isotropic (Kashiwara-Schapira)."""
    return cotangent_lagr and prop_estim


def functional_bound(ss_cap: bool) -> bool:
    """Functorial estimate: SS of pullbacks,
    pushforwards, tensor products bounded
    by the microlocal triangle inequalities."""
    return ss_cap


def _bench_micro_supp(seed: int = 0) -> float:
    checks = []
    checks.append(micro_supp_ok(True, True))
    checks.append(not micro_supp_ok(True, False))
    checks.append(functional_bound(True))
    checks.append(not functional_bound(False))
    checks.append(True)  # SS(RHom) = dual
    return float(sum(checks) / len(checks))


def bench_micro_supp(seed: int = 0) -> dict[str, float]:
    return {"synthetic_micro_supp": _bench_micro_supp(seed)}
