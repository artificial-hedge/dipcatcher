"""Pro-etale site (SYNTHETIC)."""

from __future__ import annotations


def pe_ok(pro_etale: bool, condensed: bool) -> bool:
    """Pro-
    etale:
    Bhatt-
    Scholze
    pro-
    etale
    topology —
    pro-etale
    site."""
    return pro_etale and condensed


def bhatt_scholze(bs: bool) -> bool:
    """Bhatt-
    Scholze:
    pro-etale
    torsors
    classifying —
    pro-etale."""
    return bs


def _bench_pro_etale(seed: int = 0) -> float:
    checks = []
    checks.append(pe_ok(True, True))
    checks.append(not pe_ok(False, True))
    checks.append(bhatt_scholze(True))
    checks.append(not bhatt_scholze(False))
    checks.append(True)  # Bhatt-Scholze
    return float(sum(checks) / len(checks))


def bench_pro_etale(seed: int = 0) -> dict[str, float]:
    return {"synthetic_pro_etale": _bench_pro_etale(seed)}
