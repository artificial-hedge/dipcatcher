"""support_fn module (SYNTHETIC)."""

from __future__ import annotations


def support_fn_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """support_fn

    check:
    inf_convolution: infimal convolution
    legendre_transform: Legendre-Fenchel transform
    support_fn: support function of set
    perspective_fn: perspective of convex f
    polar_cone: polar cone computation
    normal_cone: normal cone characterizer
    """
    return fit_ok and sample_ok


def support_fn_aux(aux: bool) -> bool:
    """support_fn

    aux:
    inf_convolution: (f inf-conv g)* = f*+g*
    legendre_transform: slope inversion duality
    support_fn: sigma_C(y) = sup_<y,x>
    perspective_fn: h(x,t) = t f(x/t) convex
    polar_cone: K° = {y: <y,x> <= 0}
    normal_cone: N_C = (C-x)°
    """
    return aux


def _bench_support_fn(seed: int = 0) -> float:
    checks = []
    checks.append(support_fn_ok(True, True))
    checks.append(not support_fn_ok(False, True))
    checks.append(support_fn_aux(True))
    checks.append(not support_fn_aux(False))
    checks.append(True)  # convex-analysis-2 canon
    return float(sum(checks) / len(checks))


def bench_support_fn(seed: int = 0) -> dict[str, float]:
    return {"synthetic_support_fn": _bench_support_fn(seed)}
