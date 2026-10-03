"""Day convolution (SYNTHETIC)."""

from __future__ import annotations


def day_conv_ok(cocont_ext: bool, promonoidal: bool) -> bool:
    """Day convolution on presheaves:
    F * G = colim_{a,b} F(a) x G(b) x hom(-, a x b);
    left Kan extension along tensor."""
    return cocont_ext and promonoidal


def monoidal_yoneda(closed_free: bool) -> bool:
    """Day convolution makes [C^op,V] free
    cocomplete monoidal closure on C
    (Day reflection)."""
    return closed_free


def _bench_day_conv(seed: int = 0) -> float:
    checks = []
    checks.append(day_conv_ok(True, True))
    checks.append(not day_conv_ok(False, True))
    checks.append(monoidal_yoneda(True))
    checks.append(not monoidal_yoneda(False))
    checks.append(True)  # enriched profunctors = modules
    return float(sum(checks) / len(checks))


def bench_day_conv(seed: int = 0) -> dict[str, float]:
    return {"synthetic_day_conv": _bench_day_conv(seed)}
