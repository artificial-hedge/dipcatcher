"""Toy univalence: isomorphisms between finite types induce paths (SYNTHETIC).

On a small universe of finite sets, ua : (A ≃ B) -> (A = B) maps each
bijection to an identity. We check the toy direction: equivalent
carriers (same cardinality, explicit bijection) get identified, and the
induced transport along ua respects the bijection.
"""

from __future__ import annotations

_SEED = 20261231 + 1055


def is_equiv(f: dict, a: list, b: list) -> bool:
    """f is a bijection a -> b on finite carriers."""
    if set(f.keys()) != set(a):
        return False
    return set(f.values()) == set(b) and len(set(f.values())) == len(f)


def ua(f: dict, a: list, b: list) -> tuple:
    """univalence axiom (toy): bijection -> type-level path tag."""
    if not is_equiv(f, a, b):
        raise ValueError("not an equivalence")
    return ("ua", frozenset(a), frozenset(b), tuple(sorted(f.items(), key=lambda kv: kv[0])))


def transport_ua(path: tuple, x):
    """Transport along ua path applies the underlying bijection."""
    if path[0] != "ua":
        return x
    f = dict(path[3])
    return f[x]


def univalence_injective(p: tuple, q: tuple) -> bool:
    """Two ua-paths equal only if underlying maps agree pointwise."""
    return dict(p[3]) == dict(q[3])


def bench_univalence_toy(seed: int = _SEED) -> dict[str, float]:
    del seed
    checks: list[bool] = []
    a = [0, 1]
    b = ["x", "y"]
    f = {0: "x", 1: "y"}
    checks.append(is_equiv(f, a, b))
    p = ua(f, a, b)
    checks.append(transport_ua(p, 0) == "x" and transport_ua(p, 1) == "y")
    # non-bijection rejected
    g = {0: "x", 1: "x"}
    checks.append(not is_equiv(g, a, b))
    try:
        ua(g, a, b)
        checks.append(False)
    except ValueError:
        checks.append(True)
    # different bijection -> different path
    f2 = {0: "y", 1: "x"}
    p2 = ua(f2, a, b)
    checks.append(not univalence_injective(p, p2))
    checks.append(transport_ua(p2, 0) == "y")
    return {"synthetic_univalence_toy": float(sum(checks)) / len(checks)}
