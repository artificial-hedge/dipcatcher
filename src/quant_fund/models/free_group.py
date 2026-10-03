"""Free group F(a,b): reduced-word multiplication (SYNTHETIC)."""

from __future__ import annotations


def reduce_word(w: tuple[str, ...]) -> tuple[str, ...]:
    out: list[str] = []
    inv = {"a": "A", "A": "a", "b": "B", "B": "b"}
    for g in w:
        if out and out[-1] == inv[g]:
            out.pop()
        else:
            out.append(g)
    return tuple(out)


def fmul(w1: tuple[str, ...], w2: tuple[str, ...]) -> tuple[str, ...]:
    return reduce_word(w1 + w2)


def finv(w: tuple[str, ...]) -> tuple[str, ...]:
    inv = {"a": "A", "A": "a", "b": "B", "B": "b"}
    return tuple(inv[g] for g in reversed(w))


def is_free_pair(w1: tuple[str, ...], w2: tuple[str, ...]) -> bool:
    """Nontrivial reduced words are distinct (freeness: no relations)."""
    return w1 != w2


def _bench_free_group(seed: int = 0) -> float:
    checks = []
    checks.append(reduce_word(("a", "A", "b")) == ("b",))
    checks.append(reduce_word(("a", "b", "B", "A")) == ())
    checks.append(fmul(("a", "b"), ("B", "b")) == ("a", "b"))
    checks.append(fmul(("a",), ("A",)) == ())
    checks.append(finv(("a", "b")) == ("B", "A"))
    checks.append(fmul(("a", "b"), finv(("a", "b"))) == ())
    # no relations: ab != ba in F2
    checks.append(is_free_pair(("a", "b"), ("b", "a")))
    checks.append(fmul(("a", "b"), ("a",)) == ("a", "b", "a"))
    return float(sum(checks) / len(checks))


def bench_free_group(seed: int = 0) -> dict[str, float]:
    return {"synthetic_free_group": _bench_free_group(seed)}
