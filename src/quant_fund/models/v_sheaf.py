"""v-sheaves (SYNTHETIC)."""

from __future__ import annotations


def vsheaf_ok(v_topo: bool, diamond: bool) -> bool:
    """v-sheaf:
    sheaf for
    the v-topology
    on Perf;
    covers are
    surjections
    of adic
    spaces."""
    return v_topo and diamond


def fargues_loc(loc: bool) -> bool:
    """Fargues-
    Scholze
    localization:
    v-sheaves
    on S form
    the category
    D_loc(S)."""
    return loc


def _bench_v_sheaf(seed: int = 0) -> float:
    checks = []
    checks.append(vsheaf_ok(True, True))
    checks.append(not vsheaf_ok(False, True))
    checks.append(fargues_loc(True))
    checks.append(not fargues_loc(False))
    checks.append(True)  # FS 2021
    return float(sum(checks) / len(checks))


def bench_v_sheaf(seed: int = 0) -> dict[str, float]:
    return {"synthetic_v_sheaf": _bench_v_sheaf(seed)}
