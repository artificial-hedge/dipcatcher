"""Left Kan extension along subcategory inclusion (SYNTHETIC)."""

from __future__ import annotations


def left_kan_size(colimit_contrib: dict[str, int]) -> int:
    """Lan_F G at an object = colimit of G over the comma (F down c);
    on discrete commas it is the sum of fiber sizes."""
    return sum(colimit_contrib.values())


def pointwise_kan(fiber_sizes: dict[str, int]) -> int:
    """Pointwise left Kan extension value on finite discrete data."""
    return left_kan_size(fiber_sizes)


def _bench_kan_extension(seed: int = 0) -> float:
    checks = []
    # extension of a 2-elt set along inclusion: coproducts sum fibers
    checks.append(pointwise_kan({"x": 1, "y": 1}) == 2)
    # empty fiber gives empty colimit
    checks.append(pointwise_kan({}) == 0)
    # multiplication by fiber size |F^{-1}(c)| x |G|
    checks.append(pointwise_kan({"a": 2, "b": 3}) == 5)
    # single fiber = G value itself
    checks.append(pointwise_kan({"only": 7}) == 7)
    # additivity over disjoint comma components
    checks.append(
        pointwise_kan({"a": 1, "b": 2}) == pointwise_kan({"a": 1}) + pointwise_kan({"b": 2})
    )
    return float(sum(checks) / len(checks))


def bench_kan_extension(seed: int = 0) -> dict[str, float]:
    return {"synthetic_kan_extension": _bench_kan_extension(seed)}
