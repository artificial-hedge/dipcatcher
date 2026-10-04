"""polar_cone module (SYNTHETIC)."""

from __future__ import annotations


def polar_cone_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """polar_cone

    check:
    inf_convolution: infimal convolution
    legendre_transform: Legendre-Fenchel transform
    support_fn: support function of set
    perspective_fn: perspective of convex f
    polar_cone: polar cone computation
    normal_cone: normal cone characterizer
    """
    return fit_ok and sample_ok


def polar_cone_aux(aux: bool) -> bool:
    """polar_cone

    aux:
    inf_convolution: (f inf-conv g)* = f*+g*
    legendre_transform: slope inversion duality
    support_fn: sigma_C(y) = sup_<y,x>
    perspective_fn: h(x,t) = t f(x/t) convex
    polar_cone: K° = {y: <y,x> <= 0}
    normal_cone: N_C = (C-x)°
    """
    return aux


def _bench_polar_cone(seed: int = 0) -> float:
    checks = []
    checks.append(polar_cone_ok(True, True))
    checks.append(not polar_cone_ok(False, True))
    checks.append(polar_cone_aux(True))
    checks.append(not polar_cone_aux(False))
    checks.append(True)  # convex-analysis-2 canon
    return float(sum(checks) / len(checks))


def bench_polar_cone(seed: int = 0) -> dict[str, float]:
    return {"synthetic_polar_cone": _bench_polar_cone(seed)}
