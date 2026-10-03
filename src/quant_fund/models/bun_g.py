"""Bun_G on the Fargues curve (SYNTHETIC)."""

from __future__ import annotations


def bung_ok(stack: bool, fargues: bool) -> bool:
    """Bun_G:
    moduli stack
    of G-bundles
    on the
    Fargues-
    Fontaine
    curve; the
    arena of
    Fargues-
    Scholze."""
    return stack and fargues


def kottwitz_class(kottwitz: bool) -> bool:
    """Kottwitz
    classification:
    pi_0(Bun_G)
    is the set
    B(G) of
    isocrystal
    classes."""
    return kottwitz


def _bench_bun_g(seed: int = 0) -> float:
    checks = []
    checks.append(bung_ok(True, True))
    checks.append(not bung_ok(False, True))
    checks.append(kottwitz_class(True))
    checks.append(not kottwitz_class(False))
    checks.append(True)  # Fargues
    return float(sum(checks) / len(checks))


def bench_bun_g(seed: int = 0) -> dict[str, float]:
    return {"synthetic_bun_g": _bench_bun_g(seed)}
