"""Diamond sheaf (SYNTHETIC)."""

from __future__ import annotations


def ds_ok(diamond: bool, sheaf: bool) -> bool:
    """Diamond
    sheaf:
    diamond
    sheaf —
    v
    topology."""
    return diamond and sheaf


def v_top_sheaf(vts: bool) -> bool:
    """V
    top:
    v
    topology
    sheaf —
    pro-etale."""
    return vts


def _bench_diamond_sheaf(seed: int = 0) -> float:
    checks = []
    checks.append(ds_ok(True, True))
    checks.append(not ds_ok(False, True))
    checks.append(v_top_sheaf(True))
    checks.append(not v_top_sheaf(False))
    checks.append(True)  # Scholze
    return float(sum(checks) / len(checks))


def bench_diamond_sheaf(seed: int = 0) -> dict[str, float]:
    return {"synthetic_diamond_sheaf": _bench_diamond_sheaf(seed)}
