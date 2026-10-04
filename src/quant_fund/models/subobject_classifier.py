"""Subobject classifier Ω={0,1} in FinSet via characteristic functions (SYNTHETIC)."""

from __future__ import annotations


def characteristic(s: list[int], a: int) -> tuple[int, ...]:
    """χ_S: A -> {0,1} for subobject S ⊆ A."""
    st = set(s)
    return tuple(1 if x in st else 0 for x in range(a))


def classify_pullback(s: list[int], a: int) -> list[int]:
    """Pullback of `true` (the point 1 in Ω) along χ_S is S itself."""
    chi = characteristic(s, a)
    return [x for x in range(a) if chi[x] == 1]


def is_mono(i: tuple[int, ...]) -> bool:
    """Injective = monic in FinSet."""
    return len(set(i)) == len(i)


def _bench_subobject_classifier(seed: int = 0) -> float:
    checks = []
    checks.append(characteristic([1, 3], 5) == (0, 1, 0, 1, 0))
    checks.append(classify_pullback([1, 3], 5) == [1, 3])
    checks.append(is_mono((0, 2, 1)))
    checks.append(not is_mono((0, 0, 1)))
    # every mono's classifier pulls back to itself: test several subsets
    for s in ([0], [2, 4], [], [0, 1, 2]):
        checks.append(classify_pullback(list(s), 5) == list(s))
    return sum(checks) / len(checks)


def bench_subobject_classifier(seed: int = 0) -> dict[str, float]:
    return {"synthetic_subobject_classifier": _bench_subobject_classifier(seed)}
