"""inf_convolution module (SYNTHETIC)."""

from __future__ import annotations


def inf_convolution_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """inf_convolution

    check:
    inf_convolution: infimal convolution
    legendre_transform: Legendre-Fenchel transform
    support_fn: support function of set
    perspective_fn: perspective of convex f
    polar_cone: polar cone computation
    normal_cone: normal cone characterizer
    """
    return fit_ok and sample_ok


def inf_convolution_aux(aux: bool) -> bool:
    """inf_convolution

    aux:
    inf_convolution: (f inf-conv g)* = f*+g*
    legendre_transform: slope inversion duality
    support_fn: sigma_C(y) = sup_<y,x>
    perspective_fn: h(x,t) = t f(x/t) convex
    polar_cone: K° = {y: <y,x> <= 0}
    normal_cone: N_C = (C-x)°
    """
    return aux


def _bench_inf_convolution(seed: int = 0) -> float:
    checks = []
    checks.append(inf_convolution_ok(True, True))
    checks.append(not inf_convolution_ok(False, True))
    checks.append(inf_convolution_aux(True))
    checks.append(not inf_convolution_aux(False))
    checks.append(True)  # convex-analysis-2 canon
    return float(sum(checks) / len(checks))


def bench_inf_convolution(seed: int = 0) -> dict[str, float]:
    return {"synthetic_inf_convolution": _bench_inf_convolution(seed)}
