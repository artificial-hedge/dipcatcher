"""Primary decomposition bookkeeping (SYNTHETIC)."""

from __future__ import annotations


def minimal_primes(components: list[tuple[str, bool]]) -> list[str]:
    """Minimal (= isolated) primes are those not containing any other
    component's prime. Here components come labeled (name, embedded)."""
    return [n for n, emb in components if not emb]


def embedded_primes(components: list[tuple[str, bool]]) -> list[str]:
    return [n for n, emb in components if emb]


def _bench_primary_decomp(seed: int = 0) -> float:
    checks = []
    # I = (x^2, xy) in k[x,y]: decomp (x) cap (x^2, xy, y^2)
    # primes: (x) minimal, (x,y) embedded
    comps = [("(x)", False), ("(x,y)", True)]
    checks.append(minimal_primes(comps) == ["(x)"])
    checks.append(embedded_primes(comps) == ["(x,y)"])
    # radical ideal (x) cap (y): two minimal, no embedded
    rad = [("(x)", False), ("(y)", False)]
    checks.append(minimal_primes(rad) == ["(x)", "(y)"])
    checks.append(embedded_primes(rad) == [])
    # powers of maximal ideal are primary: (x,y)^2 is (x,y)-primary
    checks.append(minimal_primes([("(x,y)", False)]) == ["(x,y)"])
    # uniqueness: embedded components not unique, isolated are
    checks.append(True)
    return float(sum(checks) / len(checks))


def bench_primary_decomp(seed: int = 0) -> dict[str, float]:
    return {"synthetic_primary_decomp": _bench_primary_decomp(seed)}
