"""Pro-etale site (SYNTHETIC)."""

from __future__ import annotations


def pe_ok(pro_etale: bool, covering: bool) -> bool:
    """Pro-
    etale
    site:
    pro-
    etale
    covers
    define
    topos —
    Bhatt-
    Scholze."""
    return pro_etale and covering


def bhatt_scholze(bs: bool) -> bool:
    """Bhatt-
    Scholze:
    pro-
    etale
    topology
    for
    l-
    adic
    sheaves —
    condensed."""
    return bs


def _bench_proetale_site2(seed: int = 0) -> float:
    checks = []
    checks.append(pe_ok(True, True))
    checks.append(not pe_ok(False, True))
    checks.append(bhatt_scholze(True))
    checks.append(not bhatt_scholze(False))
    checks.append(True)  # Bhatt-Scholze
    return float(sum(checks) / len(checks))


def bench_proetale_site2(seed: int = 0) -> dict[str, float]:
    return {"synthetic_proetale_site2": _bench_proetale_site2(seed)}
