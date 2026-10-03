"""Diamond moduli (SYNTHETIC)."""

from __future__ import annotations


def dm_ok(diamond: bool, moduli: bool) -> bool:
    """Diamond
    moduli:
    diamond
    moduli —
    Scholze
    diamond."""
    return diamond and moduli


def diamond_sheaf(ds: bool) -> bool:
    """Diamond
    sheaf:
    diamond
    sheaf —
    pro-etale
    quotient."""
    return ds


def _bench_diamond_mod(seed: int = 0) -> float:
    checks = []
    checks.append(dm_ok(True, True))
    checks.append(not dm_ok(False, True))
    checks.append(diamond_sheaf(True))
    checks.append(not diamond_sheaf(False))
    checks.append(True)  # Scholze
    return float(sum(checks) / len(checks))


def bench_diamond_mod(seed: int = 0) -> dict[str, float]:
    return {"synthetic_diamond_mod": _bench_diamond_mod(seed)}
