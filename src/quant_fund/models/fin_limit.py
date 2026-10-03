"""Finite limits in FinSet: products, equalizers, pullbacks + universal property (SYNTHETIC)."""

from __future__ import annotations

Map = tuple[int, ...]  # morphism n -> m as image list


def compose(f: Map, g: Map) -> Map:
    """g ∘ f."""
    return tuple(g[x] for x in f)


def product(a: int, b: int) -> tuple[int, Map, Map]:
    """A×B encoded as a*b elements; p1,p2 projections."""
    p1 = tuple(i // b for i in range(a * b))
    p2 = tuple(i % b for i in range(a * b))
    return a * b, p1, p2


def mediating(c: int, f: Map, g: Map, b: int) -> Map:
    """C -> A×B induced by f:C->A, g:C->B."""
    return tuple(f[x] * b + g[x] for x in range(c))


def equalizer(f: Map, g: Map) -> tuple[list[int], Map]:
    """{x | f(x)=g(x)} with inclusion."""
    elems = [x for x in range(len(f)) if f[x] == g[x]]
    inc = tuple(elems)
    return elems, inc


def pullback(f: Map, g: Map, a: int, b: int) -> tuple[int, Map, Map]:
    """A ×_C B = {(a,b)|f(a)=g(b)} encoded a*b."""
    pairs = [(x, y) for x in range(a) for y in range(b) if f[x] == g[y]]
    p1 = tuple(x for x, _ in pairs)
    p2 = tuple(y for _, y in pairs)
    return len(pairs), p1, p2


def pullback_mediating(f: Map, g: Map, a: int, b: int, z: int, u: Map, v: Map) -> Map:
    """Z -> A×_C B induced by u:Z->A, v:Z->B with f∘u = g∘v."""
    pairs = [(x, y) for x in range(a) for y in range(b) if f[x] == g[y]]
    idx = {p: i for i, p in enumerate(pairs)}
    return tuple(idx[(u[i], v[i])] for i in range(z))


def _bench_fin_limit(seed: int = 0) -> float:
    checks = []
    # product universal property: C=4, f:C->A(3), g:C->B(2)
    f = (0, 1, 2, 0)
    g = (1, 0, 1, 0)
    ab, p1, p2 = product(3, 2)
    m = mediating(4, f, g, 2)
    checks.append(compose(m, p1) == f and compose(m, p2) == g)
    checks.append(ab == 6)
    # equalizer: f,g: 3->2; f=(0,1,0) g=(0,1,1) -> eq elems {0,1}
    elems, inc = equalizer((0, 1, 0), (0, 1, 1))
    checks.append(elems == [0, 1])
    checks.append(compose(inc, (0, 1, 0)) == compose(inc, (0, 1, 1)))
    # pullback of f:2->2 f=(0,0), g:3->2 g=(1,0,1): pairs with f(x)==g(y)
    n, p1, p2 = pullback((0, 0), (1, 0, 1), 2, 3)
    checks.append(n == 2)  # x=0,1 both f=0; y=1 only g=0 -> pairs (0,1),(1,1)
    checks.append(compose(p1, (0, 0)) == compose(p2, (1, 0, 1)))
    # mediating into pullback: Z=1 u=(0) v=(1)
    mm = pullback_mediating((0, 0), (1, 0, 1), 2, 3, 1, (0,), (1,))
    checks.append(compose(mm, p1) == (0,) and compose(mm, p2) == (1,))
    return sum(checks) / len(checks)


def bench_fin_limit(seed: int = 0) -> dict[str, float]:
    return {"synthetic_fin_limit": _bench_fin_limit(seed)}
