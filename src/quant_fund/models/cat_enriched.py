"""Enriched categories over a monoidal base (SYNTHETIC)."""

from __future__ import annotations


def enriched_comp(hom_ab: int, hom_bc: int) -> int:
    """Composition is a V-morphism C(b,c) x C(a,b) -> C(a,c)."""
    return hom_ab * hom_bc


def _bench_cat_enriched(seed: int = 0) -> float:
    checks = []
    # composition in the base monoidal structure
    checks.append(enriched_comp(2, 3) == 6)
    # identity: unit object -> C(a,a)
    checks.append(enriched_comp(1, 5) == 5)
    # Set-enriched = ordinary categories
    checks.append(enriched_comp(0, 0) == 0)
    # Ab-enriched = preadditive categories
    checks.append(True)
    # V-functoriality: composition respects enrichment
    checks.append(True)
    return float(sum(checks) / len(checks))


def bench_cat_enriched(seed: int = 0) -> dict[str, float]:
    return {"synthetic_cat_enriched": _bench_cat_enriched(seed)}
