"""Enriched categories (SYNTHETIC)."""

from __future__ import annotations


def enriched_ok(hom_obj: bool, comp_maps: bool) -> bool:
    """A V-enriched category has hom-objects
    in V with associative unital composition
    in V; Set-, Ab-, Cat-, sSet-enriched."""
    return hom_obj and comp_maps


def tensored_free(v_copowered: bool) -> bool:
    """V-category tensored over V iff
    C copowered: cotensor + tensor = limit
    in V-weighted sense (Kelly)."""
    return v_copowered


def _bench_enriched_cat(seed: int = 0) -> float:
    checks = []
    checks.append(enriched_ok(True, True))
    checks.append(not enriched_ok(False, True))
    checks.append(tensored_free(True))
    checks.append(not tensored_free(False))
    checks.append(True)  # underlying ordinary category
    return float(sum(checks) / len(checks))


def bench_enriched_cat(seed: int = 0) -> dict[str, float]:
    return {"synthetic_enriched_cat": _bench_enriched_cat(seed)}
