"""Picard variety of an abelian variety (SYNTHETIC)."""

from __future__ import annotations


def pic_dim(genus: int) -> int:
    """dim Pic^0(C) = genus C."""
    return genus


def elliptic_self_dual() -> bool:
    """Elliptic curve is its own Picard variety: Pic^0(E) ~ E."""
    return True


def _bench_picard_variety(seed: int = 0) -> float:
    checks = []
    checks.append(pic_dim(3) == 3)
    checks.append(elliptic_self_dual())
    # Jacobian of genus-2 curve: dim 2
    checks.append(pic_dim(2) == 2)
    # Pic^0 of P^1 is trivial (dim 0)
    checks.append(pic_dim(0) == 0)
    # Albanese variety of C is Jac(C), dim g
    checks.append(pic_dim(4) == 4)
    return float(sum(checks) / len(checks))


def bench_picard_variety(seed: int = 0) -> dict[str, float]:
    return {"synthetic_picard_variety": _bench_picard_variety(seed)}
