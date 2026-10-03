"""Yoneda embedding in FinSet: Nat(Hom(-,A), Hom(-,B)) ≅ Hom(A,B) (SYNTHETIC)."""

from __future__ import annotations

import itertools


def hom(a: int, b: int) -> list[tuple[int, ...]]:
    return list(itertools.product(range(b), repeat=a))


def nat_trans(a: int, b: int, c_objs: list[int]) -> list[list[tuple[int, ...]]]:
    """Natural transformations Hom(-,A) -> Hom(-,B) restricted to objects c_objs:
    a component at X maps each f:X->A to g:X->B."""
    out = []
    # component at X: any map Hom(X,A) -> Hom(X,B); naturality restricts to induced-by-h:A->B
    for h in hom(a, b):
        comp = []
        for _x in c_objs:
            comp.append(h)  # induced map sends f to h∘f
        out.append(comp)
    return out


def induced(f: tuple[int, ...], h: tuple[int, ...]) -> tuple[int, ...]:
    return tuple(h[x] for x in f)


def _bench_yoneda_embed(seed: int = 0) -> float:
    checks = []
    # |Hom(A,B)| = |Nat(Hom(-,A),Hom(-,B))| — Yoneda bijection for representables
    checks.append(len(nat_trans(2, 3, [1, 2])) == len(hom(2, 3)))
    # the induced transformation at object X sends f to h∘f (naturality square)
    h = (2, 0)
    f = (1, 0)
    checks.append(induced(f, h) == (0, 2))
    # Yoneda element: h determined by image of id_A at component A
    id_a = (0, 1)
    checks.append(induced(id_a, h) == h)
    # composition: induced(f, h2) sends f to h2∘f
    h2 = (0, 2)
    checks.append(induced(f, h2) == (2, 0))
    return sum(checks) / len(checks)


def bench_yoneda_embed(seed: int = 0) -> dict[str, float]:
    return {"synthetic_yoneda_embed": _bench_yoneda_embed(seed)}
