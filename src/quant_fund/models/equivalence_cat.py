"""Equivalence of finite categories via skeletal functors (SYNTHETIC)."""

from __future__ import annotations


def fully_faithful(
    hom_src: dict[tuple[str, str], int], hom_tgt: dict[tuple[str, str], int], f: dict[str, str]
) -> bool:
    """F is fully faithful iff every hom-set map is a bijection:
    |Hom(x,y)| = |Hom(Fx,Fy)| on the sampled table."""
    return all(hom_tgt.get((f[x], f[y])) == n for (x, y), n in hom_src.items())


def essentially_surjective(
    objs_tgt: set[str], image: set[str], iso_pairs: set[frozenset[str]]
) -> bool:
    """Every object of D is isomorphic to one in the image of F."""
    for d in objs_tgt:
        if d in image:
            continue
        if not any(frozenset({d, i}) in iso_pairs for i in image):
            return False
    return True


def _bench_equivalence_cat(seed: int = 0) -> float:
    checks = []
    hs = {("a", "a"): 1, ("b", "b"): 1, ("a", "b"): 2, ("b", "a"): 0}
    ht = {("A", "A"): 1, ("B", "B"): 1, ("A", "B"): 2, ("B", "A"): 0}
    f = {"a": "A", "b": "B"}
    checks.append(fully_faithful(hs, ht, f))
    # dropping a map fails faithfulness
    ht2 = dict(ht)
    ht2[("A", "B")] = 1
    checks.append(not fully_faithful(hs, ht2, f))
    # essential surjectivity: D's extra object is isomorphic to image
    checks.append(essentially_surjective({"A", "B", "C"}, {"A", "B"}, {frozenset({"B", "C"})}))
    # non-isomorphic extra object fails
    checks.append(not essentially_surjective({"A", "B", "C"}, {"A", "B"}, set()))
    # identity map on a skeleton is an equivalence
    checks.append(essentially_surjective({"A"}, {"A"}, set()))
    return float(sum(checks) / len(checks))


def bench_equivalence_cat(seed: int = 0) -> dict[str, float]:
    return {"synthetic_equivalence_cat": _bench_equivalence_cat(seed)}
