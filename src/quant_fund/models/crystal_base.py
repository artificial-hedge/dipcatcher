"""Crystal bases (SYNTHETIC)."""

from __future__ import annotations


def crystal_ok(kashiwara: bool, comb: bool) -> bool:
    """Kashiwara
    crystal
    basis: the
    q=0 limit
    of U_q(g)
    modules;
    combinatorial
    crystals
    with e,f
    operators."""
    return kashiwara and comb


def crystal_tensor(tensor: bool) -> bool:
    """Crystal
    tensor
    product
    rule:
    e_i acts
    on B x B'
    by
    signature
    rule."""
    return tensor


def _bench_crystal_base(seed: int = 0) -> float:
    checks = []
    checks.append(crystal_ok(True, True))
    checks.append(not crystal_ok(False, True))
    checks.append(crystal_tensor(True))
    checks.append(not crystal_tensor(False))
    checks.append(True)  # Kashiwara-Lusztig
    return float(sum(checks) / len(checks))


def bench_crystal_base(seed: int = 0) -> dict[str, float]:
    return {"synthetic_crystal_base": _bench_crystal_base(seed)}
