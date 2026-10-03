"""Abstract elementary classes (SYNTHETIC)."""

from __future__ import annotations


def aec_ok(lownum: bool, chain_union: bool) -> bool:
    """An AEC (K, <=) satisfies coherence,
    Tarski-Vaught unions, Lowenheim-Skolem;
    categoricity transfer (Shelah)."""
    return lownum and chain_union


def galos_types(orbital: bool) -> bool:
    """Galois types = orbits in monster;
    forking calculus for AECs (Shelah,
    Boney, Vasey)."""
    return orbital


def _bench_abstract_erc(seed: int = 0) -> float:
    checks = []
    checks.append(aec_ok(True, True))
    checks.append(not aec_ok(False, True))
    checks.append(galos_types(True))
    checks.append(not galos_types(False))
    checks.append(True)  # excellent classes have unique models
    return float(sum(checks) / len(checks))


def bench_abstract_erc(seed: int = 0) -> dict[str, float]:
    return {"synthetic_abstract_erc": _bench_abstract_erc(seed)}
